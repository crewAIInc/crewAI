"""CrewAI platform tool builder for fetching and creating action tools."""

import logging
from types import TracebackType
from typing import Any

from crewai.tools import BaseTool

from crewai_tools.tools.crewai_platform_tools.application_selector import (
    ApplicationSelector,
)
from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)
from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    IntegrationsClient,
    LegacyClient,
)


logger = logging.getLogger(__name__)


class CrewaiPlatformToolBuilder:
    """Builds platform tools from remote action schemas."""

    def __init__(
        self,
        apps: list[str],
        integrations_client: IntegrationsClient | None = None,
    ) -> None:
        self._apps = [ApplicationSelector(app) for app in apps]
        self._integrations_client = (
            integrations_client if integrations_client is not None else LegacyClient()
        )
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
        apps = [
            f"{app.name}/{app.action}" if app.action is not None else app.name
            for app in self._apps
        ]

        try:
            response = self._integrations_client.get_actions(apps)
            response.raise_for_status()
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch platform tools for apps {apps}: {e}")
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
                app=function_details["app"],
                action_name=action_name,
                action_schema=action_schema,
                integrations_client=self._integrations_client,
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
