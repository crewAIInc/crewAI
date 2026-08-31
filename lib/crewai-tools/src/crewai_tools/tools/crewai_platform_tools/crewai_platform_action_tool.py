"""Crewai Enterprise Tools."""

import json
from typing import Any

from crewai.tools import BaseTool
from crewai.tools.tool_failure import ToolFailure
from crewai.utilities.pydantic_schema_utils import create_model_from_schema
from pydantic import Field, PrivateAttr, create_model

from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    IntegrationsClient,
    LegacyClient,
)


class CrewAIPlatformActionTool(BaseTool):
    _integrations_client: IntegrationsClient = PrivateAttr()
    app: str = Field(description="The integration slug for this action")
    action_name: str = Field(default="", description="The name of the action")
    action_schema: dict[str, Any] = Field(
        default_factory=dict, description="The schema of the action"
    )

    def __init__(
        self,
        description: str,
        app: str,
        action_name: str,
        action_schema: dict[str, Any],
        integrations_client: IntegrationsClient | None = None,
    ) -> None:
        parameters = action_schema.get("function", {}).get("parameters", {})

        if parameters and parameters.get("properties"):
            try:
                if "title" not in parameters:
                    parameters = {**parameters, "title": f"{action_name}Schema"}
                if "type" not in parameters:
                    parameters = {**parameters, "type": "object"}
                args_schema = create_model_from_schema(parameters)
            except Exception:
                args_schema = create_model(f"{action_name}Schema")
        else:
            args_schema = create_model(f"{action_name}Schema")

        super().__init__(
            name=action_name.lower().replace(" ", "_"),
            description=description,
            args_schema=args_schema,
            app=app,
        )
        self.action_name = action_name
        self.action_schema = action_schema
        self._integrations_client = (
            integrations_client if integrations_client is not None else LegacyClient()
        )

    def _run(self, **kwargs: Any) -> Any:
        try:
            cleaned_kwargs = {
                key: value for key, value in kwargs.items() if value is not None
            }

            response = self._integrations_client.execute_action(
                self.action_name, cleaned_kwargs
            )

            data = response.json()
            if not response.ok:
                if isinstance(data, dict):
                    error_info = data.get("error", {})
                    if isinstance(error_info, dict):
                        error_message = error_info.get("message", json.dumps(data))
                    else:
                        error_message = str(error_info)
                else:
                    error_message = str(data)
                # A non-2xx here means the upstream app rejected the action
                # (e.g. Slack's channel_not_found) -- report it, not prose.
                return ToolFailure(
                    message=f"API request failed: {error_message}",
                    code=str(response.status_code),
                    retryable=response.status_code >= 500,
                    details={"action": self.action_name},
                )

            return json.dumps(data, indent=2)

        except Exception as e:
            return ToolFailure(
                message=f"Error executing action {self.action_name}: {e!s}",
                code=e.__class__.__name__,
                details={"action": self.action_name},
            )
