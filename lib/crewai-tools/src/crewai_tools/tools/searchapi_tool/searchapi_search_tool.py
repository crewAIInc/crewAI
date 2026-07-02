"""SearchApi search tool for CrewAI agents."""

import logging
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import requests


logger = logging.getLogger(__name__)

BASE_URL = "https://www.searchapi.io/api/v1/search"

SUPPORTED_ENGINES = [
    "google",
    "google_news",
    "google_shopping",
    "google_jobs",
    "youtube",
    "bing",
    "baidu",
]


class SearchApiSearchToolSchema(BaseModel):
    """Input schema for SearchApi search tool."""

    search_query: str = Field(
        ..., description="Mandatory search query to perform the search."
    )
    location: str | None = Field(
        None, description="Location to perform the search from (e.g., 'New York')."
    )


class SearchApiSearchTool(BaseTool):
    """Search the internet using SearchApi.

    Supports multiple engines including Google, Google News, Google Shopping,
    Google Jobs, YouTube, Bing, and Baidu. Configure the engine at initialization.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "SearchApi Search"
    description: str = (
        "A tool that searches the internet using SearchApi. "
        "Supports multiple engines: google, google_news, google_shopping, "
        "google_jobs, youtube, bing, and baidu."
    )
    args_schema: type[BaseModel] = SearchApiSearchToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SEARCHAPI_API_KEY",
                description="API key for SearchApi (https://www.searchapi.io)",
                required=True,
            ),
        ]
    )

    engine: str = "google"
    n_results: int = 10
    country: str | None = None
    language: str | None = None

    _api_key: str | None = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the SearchApi tool and validate configuration."""
        super().__init__(**kwargs)
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError(
                f"Invalid engine: {self.engine}. "
                f"Must be one of: {', '.join(SUPPORTED_ENGINES)}"
            )
        api_key = os.getenv("SEARCHAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing SEARCHAPI_API_KEY. Get your key at https://www.searchapi.io"
            )
        self._api_key = api_key

    def _run(self, **kwargs: Any) -> Any:
        """Execute a search query against the configured SearchApi engine."""
        search_query: str | None = kwargs.get("search_query") or kwargs.get("query")
        if not search_query:
            raise ValueError("search_query is required")

        params: dict[str, Any] = {
            "engine": self.engine,
            "q": search_query,
            "num": self.n_results,
        }

        location = kwargs.get("location")
        if location:
            params["location"] = location
        if self.country:
            params["gl"] = self.country
        if self.language:
            params["hl"] = self.language

        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            response = requests.get(
                BASE_URL, params=params, headers=headers, timeout=30
            )
            response.raise_for_status()
            results: dict[str, Any] = response.json()
        except requests.RequestException as e:
            error_msg = f"An error occurred while performing the search: {e!s}"
            logger.error(error_msg)
            return error_msg

        for key in ["search_metadata", "search_parameters", "pagination"]:
            results.pop(key, None)

        return results
