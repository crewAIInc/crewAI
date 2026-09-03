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
    ToolExecutionFailure,
    ToolInfo,
)


class CrewAIPlatformActionTool(BaseTool):
    _client: IntegrationsClient = PrivateAttr()
    _tool_info: ToolInfo = PrivateAttr()
    app: str = Field(description="The integration slug for this action")

    def __init__(
        self,
        tool_info: ToolInfo,
        client: IntegrationsClient | None = None,
    ) -> None:
        schema_name = f"{tool_info.qualified_name}Schema"
        parameters = tool_info.parameters

        if parameters and parameters.get("properties"):
            try:
                if "title" not in parameters:
                    parameters = {**parameters, "title": schema_name}
                if "type" not in parameters:
                    parameters = {**parameters, "type": "object"}
                args_schema = create_model_from_schema(parameters)
            except Exception:
                args_schema = create_model(schema_name)
        else:
            args_schema = create_model(schema_name)

        super().__init__(
            name=tool_info.qualified_name,
            description=tool_info.description,
            args_schema=args_schema,
            app=tool_info.app,
        )
        self._client = client if client is not None else LegacyClient()
        self._tool_info = tool_info

    def _run(self, **kwargs: Any) -> str | ToolFailure:
        try:
            cleaned_kwargs = {
                key: value for key, value in kwargs.items() if value is not None
            }

            result = self._client.execute_action(self._tool_info, cleaned_kwargs)

            if isinstance(result, ToolExecutionFailure):
                return ToolFailure(
                    message=f"API request failed: {result.message}",
                    code=result.code,
                    retryable=result.retryable,
                    details={"action": self._tool_info.action},
                )

            return json.dumps(result.output, indent=2)

        except Exception as e:
            return ToolFailure(
                message=f"Error executing action {self._tool_info.action}: {e!s}",
                code=e.__class__.__name__,
                details={"action": self._tool_info.action},
            )
