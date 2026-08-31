"""Crewai Enterprise Tools."""

import json
from typing import Any

from crewai.tools import BaseTool
from crewai.tools.tool_failure import ToolFailure
from crewai.utilities.pydantic_schema_utils import create_model_from_schema
from crewai.utilities.string_utils import sanitize_tool_name
from pydantic import Field, PrivateAttr, create_model

from crewai_tools.tools.crewai_platform_tools._client import (
    _PlatformToolInfo,
    _PlatformToolsClient,
)


class CrewAIPlatformActionTool(BaseTool):
    _client: _PlatformToolsClient = PrivateAttr()
    _tool_info: _PlatformToolInfo = PrivateAttr()
    app: str = Field(description="The integration slug for this action")

    def __init__(
        self,
        tool_info: _PlatformToolInfo,
        client: _PlatformToolsClient,
    ) -> None:
        action = tool_info.action
        parameters = tool_info.parameters

        if parameters and parameters.get("properties"):
            try:
                if "title" not in parameters:
                    parameters = {
                        **parameters,
                        "title": f"{action}Schema",
                    }
                if "type" not in parameters:
                    parameters = {**parameters, "type": "object"}
                args_schema = create_model_from_schema(parameters)
            except Exception:
                args_schema = create_model(f"{action}Schema")
        else:
            args_schema = create_model(f"{action}Schema")

        name_parts = [
            tool_info.app,
            action,
            str(tool_info.connection_id) if tool_info.connection_id else None,
        ]
        name = sanitize_tool_name("_".join(part for part in name_parts if part))

        super().__init__(
            name=name,
            description=tool_info.description,
            args_schema=args_schema,
            app=tool_info.app,
        )
        self._tool_info = tool_info
        self._client = client

    def _run(self, **kwargs: Any) -> Any:
        try:
            cleaned_kwargs = {
                key: value for key, value in kwargs.items() if value is not None
            }

            response = self._client.execute_action(
                tool_info=self._tool_info,
                arguments=cleaned_kwargs,
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
                return ToolFailure(
                    message=f"API request failed: {error_message}",
                    code=str(response.status_code),
                    retryable=response.status_code >= 500,
                    details={"action": self._tool_info.action},
                )

            return json.dumps(data, indent=2)

        except Exception as e:
            return ToolFailure(
                message=f"Error executing action {self._tool_info.action}: {e!s}",
                code=e.__class__.__name__,
                details={"action": self._tool_info.action},
            )
