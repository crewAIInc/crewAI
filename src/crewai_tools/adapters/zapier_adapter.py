import os
import logging
from typing import List

import requests
from crewai.tools import BaseTool
from pydantic import Field, create_model

ACTIONS_URL = "https://actions.zapier.com/api/v2/ai-actions"

logger = logging.getLogger(__name__)


class ZapierActionTool(BaseTool):
    """
    A tool that wraps a Zapier action
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    action_id: str = Field(description="Zapier action ID")
    api_key: str = Field(description="Zapier API key")

    def _run(self, **kwargs) -> str:
        """Execute the Zapier action"""
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}

        instructions = kwargs.pop(
            "instructions", "Execute this action with the provided parameters"
        )

        if not kwargs:
            action_params = {"instructions": instructions, "params": {}}
        else:
            formatted_params = {}
            for key, value in kwargs.items():
                formatted_params[key] = {
                    "value": value,
                    "mode": "guess",
                }
            action_params = {"instructions": instructions, "params": formatted_params}

        execute_url = f"{ACTIONS_URL}/{self.action_id}/execute/"
        response = requests.request(
            "POST", execute_url, headers=headers, json=action_params
        )

        response.raise_for_status()

        return response.json()


class ZapierActionsAdapter:
    """
    Adapter for Zapier Actions
    """

    api_key: str

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ZAPIER_API_KEY")
        if not self.api_key:
            logger.error("Zapier Actions API key is required")
            raise ValueError("Zapier Actions API key is required")

    def get_zapier_actions(self):
        headers = {
            "x-api-key": self.api_key,
        }
        response = requests.request("GET", ACTIONS_URL, headers=headers)
        response.raise_for_status()

        response_json = response.json()
        return response_json

    def tools(self) -> List[BaseTool]:
        """Convert Zapier actions to BaseTool instances"""
        actions_response = self.get_zapier_actions()
        tools = []

        for action in actions_response.get("results", []):
            tool_name = (
                action["meta"]["action_label"]
                .replace(" ", "_")
                .replace(":", "")
                .lower()
            )

            params = action.get("params", {})
            args_fields = {}

            args_fields["instructions"] = (
                str,
                Field(description="Instructions for how to execute this action"),
            )

            for param_name, param_info in params.items():
                field_type = (
                    str  # Default to string, could be enhanced based on param_info
                )
                field_description = (
                    param_info.get("description", "")
                    if isinstance(param_info, dict)
                    else ""
                )
                args_fields[param_name] = (
                    field_type,
                    Field(description=field_description),
                )

            args_schema = create_model(f"{tool_name.title()}Schema", **args_fields)

            tool = ZapierActionTool(
                name=tool_name,
                description=action["description"],
                action_id=action["id"],
                api_key=self.api_key,
                args_schema=args_schema,
            )
            tools.append(tool)

        return tools
