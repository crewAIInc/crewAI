"""Tool for searching the web with the Querit Search API."""

import os
from typing import Any, TypeAlias

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
import requests


load_dotenv()

QueritSearchResponse: TypeAlias = dict[str, Any]
QueritFilters: TypeAlias = dict[str, Any]
TIME_RANGE_PATTERN = r"^([dwmy][1-9][0-9]*|\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2})$"


class QueritSearchToolSchema(BaseModel):
    """Input for QueritSearchTool."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(..., description="The search query string.")
    count: int | None = Field(
        default=None, ge=1, description="The maximum number of results to return."
    )
    chunks_per_doc: int | None = Field(
        default=None,
        ge=1,
        le=3,
        alias="chunksPerDoc",
        description="The number of summary chunks to return per document.",
    )
    site_include: list[str] | None = Field(
        default=None,
        description="Websites to include in the search results.",
    )
    site_exclude: list[str] | None = Field(
        default=None,
        description="Websites to exclude from the search results.",
    )
    time_range: str | None = Field(
        default=None,
        pattern=TIME_RANGE_PATTERN,
        description="Time range filter, such as d7, w1, m3, y1, or a date range.",
    )
    country_include: list[str] | None = Field(
        default=None,
        description="Countries to include in the search results.",
    )
    language_include: list[str] | None = Field(
        default=None,
        description="Languages to include in the search results.",
    )


class QueritSearchTool(BaseTool):
    """Tool that searches the web using the Querit Search API."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    name: str = "Querit Search"
    description: str = (
        "A tool that performs web searches using the Querit Search API. "
        "It returns the original Querit API JSON response."
    )
    args_schema: type[BaseModel] = QueritSearchToolSchema
    search_url: str = "https://api.querit.ai/v1/search"
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("QUERIT_API_KEY"),
        description="The Querit API key. If not provided, it will be loaded from QUERIT_API_KEY.",
    )
    count: int = Field(
        default=10, ge=1, description="The maximum number of results to return."
    )
    chunks_per_doc: int | None = Field(
        default=3,
        ge=1,
        le=3,
        alias="chunksPerDoc",
        description="The number of summary chunks to return per document.",
    )
    timeout: int = Field(default=30, description="The request timeout in seconds.")
    site_include: list[str] | None = Field(
        default=None,
        description="Websites to include in the search results.",
    )
    site_exclude: list[str] | None = Field(
        default=None,
        description="Websites to exclude from the search results.",
    )
    time_range: str | None = Field(
        default=None,
        pattern=TIME_RANGE_PATTERN,
        description="Time range filter, such as d7, w1, m3, y1, or a date range.",
    )
    country_include: list[str] | None = Field(
        default=None,
        description="Countries to include in the search results.",
    )
    language_include: list[str] | None = Field(
        default=None,
        description="Languages to include in the search results.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="QUERIT_API_KEY",
                description="API key for the Querit search service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool and ensure the Querit API key is configured."""
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError(
                "QUERIT_API_KEY environment variable is required for QueritSearchTool"
            )

    def _run(self, query: str, **kwargs: Any) -> QueritSearchResponse:
        """Execute a Querit web search.

        Args:
            query: Search query string.
            **kwargs: Optional runtime overrides for search parameters.

        Returns:
            The raw Querit API JSON response.
        """
        count = kwargs.get("count") or self.count
        chunks_per_doc = kwargs.get("chunks_per_doc") or kwargs.get("chunksPerDoc")
        if chunks_per_doc is None:
            chunks_per_doc = self.chunks_per_doc
        payload: QueritSearchResponse = {"query": query, "count": count}
        if chunks_per_doc is not None:
            payload["chunksPerDoc"] = chunks_per_doc

        filters = self._build_filters(kwargs)
        if filters:
            payload["filters"] = filters

        response: requests.Response | None = None
        last_error: requests.RequestException | None = None
        for _ in range(3):
            try:
                response = requests.post(
                    self.search_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except requests.HTTPError as error:
                last_error = error
                status_code = error.response.status_code if error.response else None
                if status_code not in {429, 500, 502, 503, 504}:
                    raise
            except requests.RequestException as error:
                last_error = error
        if response is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Querit request failed without an exception")
        search_response = response.json()
        if not isinstance(search_response, dict):
            raise ValueError("Querit API response must be a JSON object")

        return dict(search_response)

    def _build_filters(self, kwargs: dict[str, Any]) -> QueritFilters:
        """Build the Querit API filters payload from flat tool parameters.

        Args:
            kwargs: Runtime parameter overrides passed to the tool.

        Returns:
            A Querit API filters mapping, or an empty mapping when no filters are set.
        """
        filters: QueritFilters = {}
        site_include = kwargs.get("site_include") or self.site_include
        site_exclude = kwargs.get("site_exclude") or self.site_exclude
        time_range = kwargs.get("time_range") or self.time_range
        country_include = kwargs.get("country_include") or self.country_include
        language_include = kwargs.get("language_include") or self.language_include

        if site_include is not None or site_exclude is not None:
            filters["sites"] = {
                key: value
                for key, value in {
                    "include": site_include,
                    "exclude": site_exclude,
                }.items()
                if value is not None
            }
        if time_range is not None:
            filters["timeRange"] = {"date": time_range}
        if country_include is not None:
            filters["geo"] = {"countries": {"include": country_include}}
        if language_include is not None:
            filters["languages"] = {"include": language_include}

        return filters
