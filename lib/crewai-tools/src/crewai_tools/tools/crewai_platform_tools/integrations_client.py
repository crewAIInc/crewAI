"""Contract and default client for platform integrations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
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
                    selector = next(
                        (
                            selector
                            for selector in selectors
                            if selector.app == app
                            and (selector.action is None or selector.action == action)
                        ),
                        None,
                    )
                    tool_infos.append(
                        ToolInfo(
                            app=app,
                            action=action,
                            connection_id=(
                                selector.connection_id if selector is not None else None
                            ),
                            description=action_data.get(
                                "description", f"Execute {action}"
                            ),
                            parameters=action_data.get("parameters", {}),
                        )
                    )

        return tool_infos

    def execute_action(
        self, tool: ToolInfo, arguments: dict[str, Any]
    ) -> requests.Response:
        """Execute an action with the given arguments."""
        plus_api = PlusAPI()
        return requests.post(
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
            verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
        )
