"""CrewAI platform action tool."""

import json
import logging
import os
from typing import Any, Protocol
from uuid import uuid4

from crewai.tools import BaseTool
from crewai.tools.tool_failure import ToolFailure
from crewai.utilities.pydantic_schema_utils import create_model_from_schema
from crewai_core.plus_api import PlusAPI
from pydantic import PrivateAttr
import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_integration_token,
)
from crewai_tools.tools.crewai_platform_tools.platform_tool import PlatformTool


logger = logging.getLogger(__name__)


def _verify_ssl() -> bool:
    return os.environ.get("CREWAI_FACTORY", "false").lower() != "true"


class PlatformToolsClient(Protocol):
    """Discover and execute remote platform tools."""

    def list_tools(self, selector: PlatformTool) -> list[PlatformTool]:
        """Return resolved tools for an application selector."""
        ...

    def execute_tool(
        self,
        tool: PlatformTool,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute one platform tool."""
        ...


class LegacyIntegrationsClient:
    """Use the legacy integrations API for platform tools."""

    def list_tools(self, selector: PlatformTool) -> list[PlatformTool]:
        """Return tools from the legacy actions endpoint."""
        actions_url = f"{PlusAPI().base_url}/crewai_plus/api/v1/integrations/actions"
        headers = {"Authorization": f"Bearer {get_platform_integration_token()}"}
        selector_value = (
            f"{selector.application}/{selector.tool}"
            if selector.tool is not None
            else selector.application
        )

        try:
            response = requests.get(
                actions_url,
                headers=headers,
                timeout=30,
                params={"apps": selector_value},
                verify=_verify_ssl(),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch platform tools for {selector_value}: {e}")
            return []

        definitions = []
        for app, actions in response.json().get("actions", {}).items():
            for action in actions:
                name = action["name"]
                definitions.append(
                    PlatformTool(
                        application=app,
                        tool=name,
                        description=action.get("description", f"Execute {name}"),
                        input_schema=action.get("parameters", {}),
                    )
                )

        return definitions

    def execute_tool(
        self,
        tool: PlatformTool,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool through the legacy actions endpoint."""
        api_url = (
            f"{PlusAPI().base_url}"
            f"/crewai_plus/api/v1/integrations/actions/{tool.tool}/execute"
        )
        headers = {
            "Authorization": f"Bearer {get_platform_integration_token()}",
            "Content-Type": "application/json",
        }
        cleaned_arguments = {
            key: value for key, value in arguments.items() if value is not None
        }
        payload = {
            "integration": cleaned_arguments if cleaned_arguments else {"_noop": True}
        }

        response = requests.post(
            url=api_url,
            headers=headers,
            json=payload,
            timeout=60,
            verify=_verify_ssl(),
        )

        data = response.json()
        if not response.ok:
            if isinstance(data, dict):
                error_info = data.get("error", {})
                error_message = (
                    error_info.get("message", json.dumps(data))
                    if isinstance(error_info, dict)
                    else str(error_info)
                )
            else:
                error_message = str(data)
            return ToolFailure(
                message=f"API request failed: {error_message}",
                code=str(response.status_code),
                retryable=response.status_code >= 500,
                details={"action": tool.tool},
            )

        return json.dumps(data, indent=2)


class ClipperClient:
    """Use the Clipper Runtime API for platform tools."""

    def __init__(
        self,
        integration_token: str | None = None,
        deployment_instance_uuid: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._integration_token = integration_token
        self._deployment_instance_uuid = deployment_instance_uuid
        self._base_url = (base_url or PlusAPI().base_url).rstrip("/")

    def list_tools(self, selector: PlatformTool) -> list[PlatformTool]:
        """Resolve a selector and return tools with their current input schemas."""
        connection_id = str(selector.connection_id) if selector.connection_id else None
        params = {"connection_id": connection_id} if connection_id else None
        tools_url = (
            f"{self._base_url}/clipper/v1/applications/{selector.application}/tools"
        )
        if selector.tool is not None:
            tools_url = f"{tools_url}/{selector.tool}"
        response = requests.get(
            tools_url,
            headers=self._headers(),
            params=params,
            timeout=30,
            verify=_verify_ssl(),
        )
        response.raise_for_status()
        data = response.json()["data"]
        tools = data if selector.tool is None else [data]
        return [
            PlatformTool(
                application=selector.application,
                tool=tool["slug"],
                description=tool["description"],
                input_schema=tool["input_schema"],
                connection_id=selector.connection_id,
            )
            for tool in tools
        ]

    def execute_tool(
        self,
        tool: PlatformTool,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool through the Clipper Runtime API."""
        payload: dict[str, Any] = {"arguments": arguments}
        if tool.connection_id:
            payload["connection_id"] = str(tool.connection_id)

        response = requests.post(
            url=(
                f"{self._base_url}/clipper/v1/applications/{tool.application}/tools/"
                f"{tool.tool}/execute"
            ),
            headers={
                **self._headers(),
                "Content-Type": "application/json",
                "Idempotency-Key": str(uuid4()),
            },
            json=payload,
            timeout=60,
            verify=_verify_ssl(),
        )
        data = response.json()
        if not response.ok:
            errors = data["errors"]
            error = errors[0]
            return ToolFailure(
                message=error["detail"],
                code=error["code"],
                retryable=response.status_code >= 500,
                details={"action": tool.tool, "errors": errors},
            )
        return json.dumps(data["data"]["output"], indent=2)

    def _headers(self) -> dict[str, str]:
        token = self._integration_token or get_platform_integration_token()
        deployment_instance_uuid = self._deployment_instance_uuid or os.getenv(
            "CREWAI_DEPLOYMENT_INSTANCE_UUID"
        )
        headers = {"Authorization": f"Bearer {token}"}
        if deployment_instance_uuid:
            headers["X-Crewai-Deployment-Instance-Id"] = deployment_instance_uuid
        return headers


class CrewAIPlatformActionTool(BaseTool):
    _client: PlatformToolsClient = PrivateAttr()
    _platform_tool: PlatformTool = PrivateAttr()

    def __init__(
        self,
        platform_tool: PlatformTool,
        client: PlatformToolsClient,
    ) -> None:
        if (
            platform_tool.tool is None
            or platform_tool.input_schema is None
            or platform_tool.description is None
        ):
            raise ValueError("Platform tool must be resolved before it can be built")
        args_schema = create_model_from_schema(
            {"type": "object", **platform_tool.input_schema},
        )

        super().__init__(
            name=platform_tool.python_identifier,
            description=platform_tool.description,
            args_schema=args_schema,
        )
        self._client = client
        self._platform_tool = platform_tool

    def _run(self, **kwargs: Any) -> Any:
        try:
            return self._client.execute_tool(self._platform_tool, kwargs)

        except Exception as e:
            return ToolFailure(
                message=f"Error executing action {self._platform_tool.tool}: {e!s}",
                code=e.__class__.__name__,
                details={"action": self._platform_tool.tool},
            )
