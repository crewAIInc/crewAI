import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field
import requests


SEARCH_URL = "https://www.searchapi.io/api/v1/search"


class SearchApiBaseTool(BaseTool):
    """Base class for SearchApi functionality with shared capabilities."""

    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SEARCHAPI_API_KEY",
                description="API key for SearchApi searches",
                required=True,
            ),
        ]
    )

    api_key: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        api_key = os.getenv("SEARCHAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key, you can get the key from https://www.searchapi.io"
            )
        self.api_key = api_key

    def _search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perform a request against the SearchApi search endpoint."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(SEARCH_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def _omit_fields(
        self, data: dict[str, Any], omit_fields: list[str]
    ) -> dict[str, Any]:
        """Return a copy of the response without noisy metadata fields."""
        return {k: v for k, v in data.items() if k not in omit_fields}
