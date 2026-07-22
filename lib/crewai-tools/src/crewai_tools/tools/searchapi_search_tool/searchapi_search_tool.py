import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests


load_dotenv()


class SearchApiSearchToolSchema(BaseModel):
    """Input schema for SearchApiSearchTool."""

    query: str = Field(..., description="The search query string.")


class SearchApiSearchTool(BaseTool):
    """A tool that performs web searches using the SearchApi.io API."""

    name: str = "SearchApi Search"
    description: str = (
        "A tool that performs web searches using the SearchApi.io API. "
        "It supports 100+ search engines (e.g. google, bing, baidu) and "
        "returns the organic results as structured JSON data."
    )
    args_schema: type[BaseModel] = SearchApiSearchToolSchema
    search_url: str = "https://www.searchapi.io/api/v1/search"
    engine: str = "google"
    n_results: int = 10
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SEARCHAPI_API_KEY"),
        description="The SearchApi.io API key. If not provided, it will be loaded from the environment variable SEARCHAPI_API_KEY.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SEARCHAPI_API_KEY",
                description="API key for SearchApi.io search service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError(
                "SEARCHAPI_API_KEY environment variable is required for SearchApiSearchTool"
            )

    def _run(self, query: str) -> str:
        """Perform a search using the SearchApi.io API.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing a list of organic results, each with the
            keys ``title``, ``link``, ``snippet``, and ``position``. Results
            missing a title or link are skipped. On a request failure, a
            human-readable error string (prefixed with ``"Error performing
            search:"``) is returned instead of raising; for HTTP errors the
            API's error message is surfaced when available.
        """
        params: dict[str, str | int] = {
            "engine": self.engine,
            "q": query,
            "num": self.n_results,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(
                self.search_url, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.HTTPError as e:
            message = str(e)
            try:
                error_detail = e.response.json().get("error")
            except (ValueError, AttributeError):
                error_detail = None
            if error_detail:
                message = f"{message} - {error_detail}"
            return f"Error performing search: {message}"
        except requests.RequestException as e:
            return f"Error performing search: {e!s}"

        results = []
        for result in data.get("organic_results", []):
            title = result.get("title")
            link = result.get("link")
            if not title or not link:
                continue
            results.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": result.get("snippet"),
                    "position": result.get("position"),
                }
            )

        return json.dumps(results, indent=2)
