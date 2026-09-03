"""Contract and default client for platform integrations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol
from uuid import UUID

from crewai.utilities.string_utils import sanitize_tool_name
from crewai_core.plus_api import PlusAPI
import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_integration_token,
)


@dataclass(frozen=True)
class ApplicationSelector:
    """Represent an application selector."""

    app: str
    action: str | None
    connection_id: UUID | None

    @classmethod
    def from_string(cls, value: str) -> ApplicationSelector:
        """Parse the ``application[/action][@connection_uuid]`` syntax.

        Raises:
            ValueError: If the selector does not follow the supported syntax.
        """
        if not value:
            raise ValueError(f"Invalid application selector {value!r}: cannot be empty")
        if "@" in value and "/" in value and value.index("@") < value.index("/"):
            raise ValueError(
                f"Invalid application selector {value!r}: "
                "connection ID must be the last segment"
            )

        app_and_action, connection_separator, connection_id = value.partition("@")
        app, action_separator, action = app_and_action.partition("/")

        if not app:
            raise ValueError(
                f"Invalid application selector {value!r}: application cannot be empty"
            )
        if action_separator and not action:
            raise ValueError(
                f"Invalid application selector {value!r}: action cannot be empty"
            )
        if connection_separator and not connection_id:
            raise ValueError(
                f"Invalid application selector {value!r}: connection ID cannot be empty"
            )

        parsed_connection_id = None
        if connection_id:
            try:
                parsed_connection_id = UUID(connection_id)
            except ValueError as error:
                raise ValueError(
                    f"Invalid application selector {value!r}: "
                    "connection ID must be a valid UUID"
                ) from error

        return cls(
            app=app,
            action=action if action_separator else None,
            connection_id=parsed_connection_id,
        )


@dataclass(frozen=True)
class ToolInfo:
    """Describe a normalized platform action."""

    app: str
    action: str
    connection_id: UUID | None
    description: str
    parameters: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        """Return the qualified tool name."""
        parts = [self.app, self.action]
        if self.connection_id is not None:
            parts.append(str(self.connection_id))
        return sanitize_tool_name("_".join(parts))


@dataclass(frozen=True)
class ToolExecutionSuccess:
    """Represent a successful platform action execution."""

    output: dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionFailure:
    """Represent an expected platform action failure."""

    message: str
    code: str
    retryable: bool


ToolExecutionResult = ToolExecutionSuccess | ToolExecutionFailure


class IntegrationsClient(Protocol):
    """Define the contract for platform integrations clients."""

    def get_actions(self, selectors: list[ApplicationSelector]) -> list[ToolInfo]:
        """Get the actions available for the selected applications."""

    def execute_action(
        self, tool: ToolInfo, arguments: dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute an action with the given arguments."""


class ClipperClient:
    """Use the Clipper platform integrations API."""

    _RESOURCE = "/clipper/v1"

    def get_actions(self, selectors: list[ApplicationSelector]) -> list[ToolInfo]:
        """Get the actions available for the selected applications."""
        plus_api = PlusAPI()
        base_url = f"{plus_api.base_url.rstrip('/')}{self._RESOURCE}"
        headers = self._headers()
        tool_infos: list[ToolInfo] = []

        for selector in selectors:
            url = f"{base_url}/applications/{selector.app}/tools"
            if selector.action is not None:
                url = f"{url}/{selector.action}"

            params = (
                {"connection_id": str(selector.connection_id)}
                if selector.connection_id is not None
                else {}
            )
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30,
                verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
            )
            response.raise_for_status()
            data = response.json()["data"]
            actions = data if selector.action is None else [data]

            tool_infos.extend(
                ToolInfo(
                    app=selector.app,
                    action=action["slug"],
                    connection_id=selector.connection_id,
                    description=action["description"],
                    parameters=action["input_schema"],
                )
                for action in actions
            )

        return tool_infos

    def execute_action(
        self, tool: ToolInfo, arguments: dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute an action with the given arguments."""
        plus_api = PlusAPI()
        payload: dict[str, Any] = {"arguments": arguments}
        if tool.connection_id is not None:
            payload["connection_id"] = str(tool.connection_id)

        response = requests.post(
            (
                f"{plus_api.base_url.rstrip('/')}{self._RESOURCE}"
                f"/applications/{tool.app}/tools/{tool.action}/execute"
            ),
            headers=self._headers(),
            json=payload,
            timeout=60,
            verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
        )
        if 200 <= response.status_code < 300:
            data = response.json()
            return ToolExecutionSuccess(output=data["data"]["output"])

        try:
            error = response.json()["errors"][0]
            message = error["detail"]
            code = error["code"]
        except (
            requests.exceptions.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ):
            message = f"Upstream API request failed with status {response.status_code}."
            code = str(response.status_code)

        return ToolExecutionFailure(
            message=message,
            code=code,
            retryable=500 <= response.status_code < 600,
        )

    @staticmethod
    def _headers() -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {get_platform_integration_token()}",
        }
        deployment_instance_uuid = os.getenv("CREWAI_DEPLOYMENT_INSTANCE_UUID")
        if deployment_instance_uuid:
            headers["X-Crewai-Deployment-Instance-Id"] = deployment_instance_uuid

        return headers


class LegacyClient:
    """Use the existing CrewAI platform integrations API."""

    def get_actions(self, selectors: list[ApplicationSelector]) -> list[ToolInfo]:
        """Get the actions available for the selected applications."""
        plus_api = PlusAPI()
        apps = [
            f"{selector.app}/{selector.action}"
            if selector.action is not None
            else selector.app
            for selector in selectors
        ]
        response = requests.get(
            f"{plus_api.base_url.rstrip('/')}{plus_api.INTEGRATIONS_RESOURCE}/actions",
            headers={"Authorization": f"Bearer {get_platform_integration_token()}"},
            timeout=30,
            params={"apps": ",".join(apps)},
            verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
        )
        response.raise_for_status()

        tool_infos: list[ToolInfo] = []
        action_categories = response.json().get("actions", {})
        for app, actions in action_categories.items():
            if not isinstance(actions, list):
                continue
            for action_data in actions:
                if not isinstance(action_data, dict):
                    continue
                if action := action_data.get("name"):
                    parameters = action_data.get("parameters", {})
                    if not isinstance(parameters, dict):
                        parameters = {}

                    tool_infos.extend(
                        ToolInfo(
                            app=app,
                            action=action,
                            connection_id=selector.connection_id,
                            description=action_data.get(
                                "description", f"Execute {action}"
                            ),
                            parameters=parameters,
                        )
                        for selector in selectors
                        if selector.app == app and selector.action in (None, action)
                    )

        return tool_infos

    def execute_action(
        self, tool: ToolInfo, arguments: dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute an action with the given arguments."""
        plus_api = PlusAPI()
        response = requests.post(
            url=(
                f"{plus_api.base_url.rstrip('/')}{plus_api.INTEGRATIONS_RESOURCE}"
                f"/actions/{tool.action}/execute"
            ),
            headers={
                "Authorization": f"Bearer {get_platform_integration_token()}",
                "Content-Type": "application/json",
            },
            json={"integration": arguments if arguments else {"_noop": True}},
            timeout=60,
            allow_redirects=False,
            verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
        )
        data = response.json()
        if not 200 <= response.status_code < 300:
            if isinstance(data, dict):
                error_info = data.get("error", {})
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", json.dumps(data))
                else:
                    error_message = str(error_info)
            else:
                error_message = str(data)

            return ToolExecutionFailure(
                message=str(error_message),
                code=str(response.status_code),
                retryable=response.status_code >= 500,
            )

        return ToolExecutionSuccess(output=data)


def client_for_selector(selector: ApplicationSelector) -> IntegrationsClient:
    """Select the integrations client for an application selector."""
    if selector.connection_id is not None:
        return ClipperClient()
    return LegacyClient()
