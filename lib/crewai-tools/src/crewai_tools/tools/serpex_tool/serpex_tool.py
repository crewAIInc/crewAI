import json
import logging
import os
from typing import Any, TypedDict

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger(__name__)


class SearchMetadata(TypedDict):
    """Metadata from search results."""

    number_of_results: int
    response_time: float
    timestamp: str
    credits_used: int


class SearchResult(TypedDict, total=False):
    """Search result data."""

    title: str
    url: str
    snippet: str
    position: int
    engine: str
    published_date: str | None


class FormattedResults(TypedDict, total=False):
    """Formatted search results from Serpex API."""

    query: str
    engines: list[str]
    metadata: SearchMetadata
    results: list[SearchResult]
    suggestions: list[str]


class SerpexToolSchema(BaseModel):
    """Input for SerpexTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class SerpexTool(BaseTool):
    name: str = "Search the internet with Serpex"
    description: str = (
        "A tool that can be used to search the internet across multiple search engines "
        "with a search_query. Supports automatic engine routing across Google, Bing, "
        "DuckDuckGo, Brave, Yahoo, and Yandex. Handles captchas and blocking automatically."
    )
    args_schema: type[BaseModel] = SerpexToolSchema
    base_url: str = "https://api.serpex.dev"
    engine: str = "auto"
    time_range: str = "all"
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPEX_API_KEY", description="API key for Serpex", required=True
            ),
        ]
    )

    def _validate_engine(self, engine: str) -> str:
        """Validate the search engine parameter."""
        engine = engine.lower()
        allowed_engines = [
            "auto",
            "google",
            "bing",
            "duckduckgo",
            "brave",
            "yahoo",
            "yandex",
        ]
        if engine not in allowed_engines:
            raise ValueError(
                f"Invalid engine: {engine}. Must be one of: {', '.join(allowed_engines)}"
            )
        return engine

    def _validate_time_range(self, time_range: str) -> str:
        """Validate the time range parameter."""
        time_range = time_range.lower()
        allowed_ranges = ["all", "day", "week", "month", "year"]
        if time_range not in allowed_ranges:
            raise ValueError(
                f"Invalid time_range: {time_range}. Must be one of: {', '.join(allowed_ranges)}"
            )
        return time_range

    def _process_search_results(
        self, results: list[dict[str, Any]]
    ) -> list[SearchResult]:
        """Process search results."""
        processed_results: list[SearchResult] = []
        for result in results:
            try:
                result_data: SearchResult = {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", 0),
                    "engine": result.get("engine", ""),
                    "published_date": result.get("published_date"),
                }
                processed_results.append(result_data)
            except KeyError:  # noqa: PERF203
                logger.warning(f"Skipping malformed search result: {result}")
                continue
        return processed_results  # type: ignore[return-value]

    def _make_api_request(
        self, search_query: str, engine: str, time_range: str
    ) -> dict[str, Any]:
        """Make API request to Serpex."""
        search_url = f"{self.base_url}/api/search"
        params = {
            "q": search_query,
            "engine": engine,
            "category": "web",
            "time_range": time_range,
            "format": "json",
        }

        headers = {
            "Authorization": f"Bearer {os.environ['SERPEX_API_KEY']}",
            "Content-Type": "application/json",
        }

        response = None
        try:
            response = requests.get(
                search_url, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                logger.error("Empty response from Serpex API")
                raise ValueError("Empty response from Serpex API")
            return results
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to Serpex API: {e}"
            if response is not None and hasattr(response, "content"):
                error_msg += f"\nResponse content: {response.content.decode('utf-8', errors='replace')}"
            logger.error(error_msg)
            raise
        except json.JSONDecodeError as e:
            if response is not None and hasattr(response, "content"):
                logger.error(f"Error decoding JSON response: {e}")
                logger.error(
                    f"Response content: {response.content.decode('utf-8', errors='replace')}"
                )
            else:
                logger.error(
                    f"Error decoding JSON response: {e} (No response content available)"
                )
            raise

    def _run(self, **kwargs: Any) -> FormattedResults:
        """Execute the search operation."""
        search_query: str | None = kwargs.get("search_query") or kwargs.get("query")
        engine: str = kwargs.get("engine", self.engine)
        time_range: str = kwargs.get("time_range", self.time_range)

        if not search_query:
            raise ValueError("search_query is required")

        # Validate parameters
        engine = self._validate_engine(engine)
        time_range = self._validate_time_range(time_range)

        # Make API request
        results = self._make_api_request(search_query, engine, time_range)

        # Format results
        formatted_results: FormattedResults = {
            "query": results.get("query", search_query),
            "engines": results.get("engines", [engine]),
            "metadata": {
                "number_of_results": results.get("metadata", {}).get(
                    "number_of_results", 0
                ),
                "response_time": results.get("metadata", {}).get("response_time", 0.0),
                "timestamp": results.get("metadata", {}).get("timestamp", ""),
                "credits_used": results.get("metadata", {}).get("credits_used", 1),
            },
            "results": self._process_search_results(results.get("results", [])),
            "suggestions": results.get("suggestions", []),
        }

        return formatted_results  # type: ignore[return-value]
