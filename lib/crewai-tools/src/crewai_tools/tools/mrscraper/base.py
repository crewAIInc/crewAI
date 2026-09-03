"""Base class and credential resolution for MrScraper tools."""

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field, PrivateAttr

from crewai_tools.tools.mrscraper.client import MrScraperClient


def resolve_api_token(api_token: str | None = None) -> str:
    """Resolve a nonblank token without exposing its value in an error."""
    token = api_token if api_token is not None else os.getenv("MRSCRAPER_API_TOKEN")
    if token is None or not token.strip():
        raise ValueError(
            "MRSCRAPER_API_TOKEN is required; set it in the environment before "
            "creating a MrScraper tool"
        )
    return token


class MrScraperBaseTool(BaseTool):
    """Base for public MrScraper tools with a private shared client."""

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="MRSCRAPER_API_TOKEN",
                description="MrScraper API token",
                required=True,
            )
        ]
    )
    _client: MrScraperClient = PrivateAttr()

    def __init__(
        self,
        *,
        api_token: str | None = None,
        client: MrScraperClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tool with an injected client or resolved API token."""
        super().__init__(**kwargs)
        self._client = client or MrScraperClient(resolve_api_token(api_token))


__all__ = ["MrScraperBaseTool", "resolve_api_token"]
