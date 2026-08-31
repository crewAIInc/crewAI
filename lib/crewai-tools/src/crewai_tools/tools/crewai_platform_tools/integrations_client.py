"""Contract and default client for platform integrations."""

import os
from typing import Any, Protocol

import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)


class IntegrationsResponse(Protocol):
    """Define the response operations used by platform tools."""

    @property
    def ok(self) -> bool:
        """Return whether the request succeeded."""

    @property
    def status_code(self) -> int:
        """Return the HTTP response status code."""

    def json(self) -> Any:
        """Decode the response body as JSON."""

    def raise_for_status(self) -> None:
        """Raise an error when the request failed."""


class IntegrationsClient(Protocol):
    """Define the client operations required by CrewAI platform tools."""

    def get_actions(self, apps: list[str]) -> IntegrationsResponse:
        """Get the actions available for the selected applications."""

    def execute_action(
        self, action_name: str, arguments: dict[str, Any]
    ) -> IntegrationsResponse:
        """Execute an action with the given arguments."""


class LegacyClient:
    """Use the existing CrewAI platform integrations API."""

    def get_actions(self, apps: list[str]) -> requests.Response:
        """Get the actions available for the selected applications."""
        return requests.get(
            f"{get_platform_api_base_url()}/actions",
            headers={"Authorization": f"Bearer {get_platform_integration_token()}"},
            timeout=30,
            params={"apps": ",".join(apps)},
            verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
        )

    def execute_action(
        self, action_name: str, arguments: dict[str, Any]
    ) -> requests.Response:
        """Execute an action with the given arguments."""
        return requests.post(
            url=f"{get_platform_api_base_url()}/actions/{action_name}/execute",
            headers={
                "Authorization": f"Bearer {get_platform_integration_token()}",
                "Content-Type": "application/json",
            },
            json={"integration": arguments if arguments else {"_noop": True}},
            timeout=60,
            verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
        )
