from typing import Any
import os
from crewai.tools import BaseTool
import requests

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)
from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)


class CrewaiPlatformToolBuilder:
    def __init__(
        self,
        apps: list[str],
    ):
        self._apps = apps
        self._actions_schema = {}  # type: ignore[var-annotated]
        self._tools = None

    def tools(self) -> list[BaseTool]:
        if self._tools is None:
            self._fetch_actions()
            self._create_tools()
        return self._tools if self._tools is not None else []

    def _fetch_actions(self):
        actions_url = f"{get_platform_api_base_url()}/actions"
        headers = {"Authorization": f"Bearer {get_platform_integration_token()}"}

        try:
            response = requests.get(
                actions_url,
                headers=headers,
                timeout=30,
                params={"apps": ",".join(self._apps)},
                verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
            )
            response.raise_for_status()
        except Exception:
            return

        raw_data = response.json()

        self._actions_schema = {}
        action_categories = raw_data.get("actions", {})

        for app, action_list in action_categories.items():
            if isinstance(action_list, list):
                for action in action_list:
                    if action_name := action.get("name"):
                        action_schema = {
                            "function": {
                                "name": action_name,
                                "description": action.get(
                                    "description", f"Execute {action_name}"
                                ),
                                "parameters": action.get("parameters", {}),
                                "app": app,
                            }
                        }
                        self._actions_schema[action_name] = action_schema

    def _generate_detailed_description(
        self, schema: dict[str, Any], indent: int = 0
    ) -> list[str]:
        descriptions = []
        indent_str = "  " * indent

        schema_type = schema.get("type", "string")

        if schema_type == "object":
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])

            if properties:
                descriptions.append(f"{indent_str}Object with properties:")
                for prop_name, prop_schema in properties.items():
                    prop_desc = prop_schema.get("description", "")
                    is_required = prop_name in required_fields
                    req_str = " (required)" if is_required else " (optional)"
                    descriptions.append(
                        f"{indent_str}  - {prop_name}: {prop_desc}{req_str}"
                    )

                    if prop_schema.get("type") == "object":
                        descriptions.extend(
                            self._generate_detailed_description(prop_schema, indent + 2)
                        )
                    elif prop_schema.get("type") == "array":
                        items_schema = prop_schema.get("items", {})
                        if items_schema.get("type") == "object":
                            descriptions.append(f"{indent_str}    Array of objects:")
                            descriptions.extend(
                                self._generate_detailed_description(
                                    items_schema, indent + 3
                                )
                            )
                        elif "enum" in items_schema:
                            descriptions.append(
                                f"{indent_str}    Array of enum values: {items_schema['enum']}"
                            )
                    elif "enum" in prop_schema:
                        descriptions.append(
                            f"{indent_str}    Enum values: {prop_schema['enum']}"
                        )

        return descriptions

    def _create_tools(self):
        tools = []

        for action_name, action_schema in self._actions_schema.items():
            function_details = action_schema.get("function", {})
            description = function_details.get("description", f"Execute {action_name}")

            parameters = function_details.get("parameters", {})
            param_descriptions = []

            if parameters.get("properties"):
                param_descriptions.append("\nDetailed Parameter Structure:")
                param_descriptions.extend(
                    self._generate_detailed_description(parameters)
                )

            full_description = description + "\n".join(param_descriptions)

            tool = CrewAIPlatformActionTool(
                description=full_description,
                action_name=action_name,
                action_schema=action_schema,
            )

            tools.append(tool)

        self._tools = tools

    def __enter__(self):
        return self.tools()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
