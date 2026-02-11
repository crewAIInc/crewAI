"""CrewAI platform tool builder for fetching and creating action tools."""

import logging
import os
from types import TracebackType
from typing import Any

from crewai.tools import BaseTool
import requests

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)
from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)


logger = logging.getLogger(__name__)


class CrewaiPlatformToolBuilder:
    """Builds platform tools from remote action schemas."""

    def __init__(
        self,
        apps: list[str],
    ) -> None:
        self._apps = apps
        self._actions_schema: dict[str, dict[str, Any]] = {}
        self._tools: list[BaseTool] | None = None

    def tools(self) -> list[BaseTool]:
        """Fetch actions and return built tools."""
        if self._tools is None:
            self._fetch_actions()
            self._create_tools()
        return self._tools if self._tools is not None else []

    def _fetch_actions(self) -> None:
        """Fetch action schemas from the platform API."""
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
        except Exception as e:
            logger.error(f"Failed to fetch platform tools for apps {self._apps}: {e}")
            return

        raw_data = response.json()

        self._actions_schema = {}
        action_categories = raw_data.get("actions", {})

        for app, action_list in action_categories.items():
            if isinstance(action_list, list):
                for action in action_list:
                    if not isinstance(action, dict):
                        continue
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

    def _create_tools(self) -> None:
        """Create tool instances from fetched action schemas."""
        tools: list[BaseTool] = []

        for action_name, action_schema in self._actions_schema.items():
            function_details = action_schema.get("function", {})
            description = function_details.get("description", f"Execute {action_name}")

            tool = CrewAIPlatformActionTool(
                description=description,
                action_name=action_name,
                action_schema=action_schema,
            )

            tools.append(tool)

        self._tools = tools

    def __enter__(self) -> list[BaseTool]:
        """Enter context manager and return tools."""
        return self.tools()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
