"""Crewai Enterprise Tools."""

import json
import os
from typing import Any

from crewai.tools import BaseTool
from crewai.utilities.pydantic_schema_utils import create_model_from_schema
from pydantic import Field, create_model
import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)


class CrewAIPlatformActionTool(BaseTool):
    action_name: str = Field(default="", description="The name of the action")
    action_schema: dict[str, Any] = Field(
        default_factory=dict, description="The schema of the action"
    )

    def __init__(
        self,
        description: str,
        action_name: str,
        action_schema: dict[str, Any],
    ):
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
        )
        self.action_name = action_name
        self.action_schema = action_schema

    def _run(self, **kwargs: Any) -> str:
        try:
            cleaned_kwargs = {
                key: value for key, value in kwargs.items() if value is not None
            }

            api_url = (
                f"{get_platform_api_base_url()}/actions/{self.action_name}/execute"
            )
            token = get_platform_integration_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {
                "integration": cleaned_kwargs if cleaned_kwargs else {"_noop": True}
            }

            response = requests.post(
                url=api_url,
                headers=headers,
                json=payload,
                timeout=60,
                verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
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
                return f"API request failed: {error_message}"

            return json.dumps(data, indent=2)

        except Exception as e:
            return f"Error executing action {self.action_name}: {e!s}"
