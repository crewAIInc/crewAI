from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any
from uuid import UUID

from crewai.plus_api import PlusAPI
import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_integration_token,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PlatformToolSelector:
    app: str
    action: str | None
    connection_id: UUID | None

    @classmethod
    def from_string(cls, value: str) -> _PlatformToolSelector:
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
class _PlatformToolInfo:
    app: str
    action: str
    connection_id: UUID | None
    description: str
    parameters: dict[str, Any]


def _should_verify_ssl() -> bool:
    return os.environ.get("CREWAI_FACTORY", "false").lower() != "true"


class _PlatformToolsClient:
    def _headers(self) -> dict[str, str]:
        token = get_platform_integration_token()
        headers = {"Authorization": f"Bearer {token}"}
        if deployment_instance_uuid := os.getenv("CREWAI_DEPLOYMENT_INSTANCE_UUID"):
            headers["X-Crewai-Deployment-Instance-Id"] = deployment_instance_uuid
        return headers

    def get_tools(
        self, selectors: list[_PlatformToolSelector]
    ) -> list[_PlatformToolInfo]:
        headers = self._headers()
        resolved_tools: list[_PlatformToolInfo] = []

        for selector in selectors:
            resolved_tools.extend(self._get_tools_for_selector(selector, headers))

        return resolved_tools

    def _get_tools_for_selector(
        self, selector: _PlatformToolSelector, headers: dict[str, str]
    ) -> list[_PlatformToolInfo]:
        try:
            url = f"{PlusAPI().base_url}/clipper/v1/applications/{selector.app}/tools"
            if selector.action is not None:
                url = f"{url}/{selector.action}"
            connection_id = (
                str(selector.connection_id) if selector.connection_id else None
            )
            response = requests.get(
                url,
                headers=headers,
                timeout=30,
                params={"connection_id": connection_id} if connection_id else None,
                verify=_should_verify_ssl(),
            )
            response.raise_for_status()
            data = response.json()["data"]
            tools = data if selector.action is None else [data]
            return [
                _PlatformToolInfo(
                    app=selector.app,
                    action=tool["slug"],
                    connection_id=selector.connection_id,
                    description=tool["description"],
                    parameters=tool["input_schema"],
                )
                for tool in tools
            ]
        except Exception as error:
            logger.error(f"Failed to fetch platform tools for {selector}: {error}")
            return []

    def execute_action(
        self, tool_info: _PlatformToolInfo, arguments: dict[str, Any]
    ) -> requests.Response:
        payload: dict[str, Any] = {"arguments": arguments}
        if tool_info.connection_id:
            payload["connection_id"] = str(tool_info.connection_id)
        return requests.post(
            url=(
                f"{PlusAPI().base_url}/clipper/v1/applications/{tool_info.app}/"
                f"tools/{tool_info.action}/execute"
            ),
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
            verify=_should_verify_ssl(),
        )
