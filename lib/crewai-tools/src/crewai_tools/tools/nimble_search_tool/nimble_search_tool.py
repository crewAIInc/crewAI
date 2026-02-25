import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
try:
    from nimble_python import AsyncNimble, Nimble  # type: ignore[import-untyped]
except ImportError:
    Nimble = Any
    AsyncNimble = Any


class NimbleSearchToolSchema(BaseModel):
    """Input schema for NimbleSearchTool.

    Runtime parameters allow agents to dynamically adjust search behavior based on task needs.
    Configuration preferences (locale, country, parsing_type) are set at tool initialization.
    """

    query: str = Field(..., description="The search query string.")
    max_results: int | None = Field(
        default=None,
        description="Maximum number of results. Agents can adjust based on task comprehensiveness (e.g., 2 for focused research, 10 for broad exploration). Defaults to tool's max_results (3).",
    )
    deep_search: bool | None = Field(
        default=None,
        description="Use True for detailed content extraction (research, analysis) or False for quick metadata searches (fact-checking, link discovery). Defaults to tool's deep_search (False).",
    )
    include_answer: bool | None = Field(
        default=None,
        description="Set to True to get an LLM-generated direct answer along with search results. Useful for question-answering tasks. Defaults to tool's include_answer (False).",
    )
    time_range: Literal["hour", "day", "week", "month", "year"] | None = Field(
        default=None,
        description="Filter results by time period. Use 'week' or 'day' for news/current events, omit for general topics. Defaults to no time filter.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Restrict search to specific domains (e.g., ['github.com', 'arxiv.org'] for technical research). Defaults to all domains.",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Exclude specific domains from results (e.g., ['pinterest.com'] to avoid image sites). Defaults to no exclusions.",
    )


class NimbleSearchTool(BaseTool):
    """Tool that uses the Nimble Search API to perform web searches.

    Configuration attributes (set at initialization):
      api_key: The Nimble API key.
      max_results: Default maximum number of results (can be overridden at runtime).
      deep_search: Default search mode - False for fast, True for deep (can be overridden at runtime).
      include_answer: Default for LLM-generated answers (can be overridden at runtime).
      parsing_type: Content format - 'markdown' (recommended), 'plain_text', or 'simplified_html'.
      locale: Locale for search results (e.g., 'en-US').
      country: Country code for geo-targeting (e.g., 'US').
      max_content_length_per_result: Maximum content length per result.

    Runtime parameters (passed when calling the tool):
      query: Search query string (required).
      max_results: Override default max_results.
      deep_search: Override default search mode.
      include_answer: Override default include_answer.
      time_range: Filter by time ('hour', 'day', 'week', 'month', 'year').
      include_domains: Restrict to specific domains.
      exclude_domains: Exclude specific domains.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Nimble | None = None
    async_client: AsyncNimble | None = None
    name: str = "Nimble Search"
    description: str = (
        "A tool that performs web searches using the Nimble Search API. "
        "It returns a JSON object containing the search results. "
        "Use deep_search=False for fast metadata-only searches (token-efficient), "
        "or deep_search=True for full content extraction with detailed parsing."
    )
    args_schema: type[BaseModel] = NimbleSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("NIMBLE_API_KEY"),
        description="The Nimble API key. If not provided, it will be loaded from the environment variable NIMBLE_API_KEY.",
    )
    max_results: int = Field(
        default=3, description="Default maximum number of results to return."
    )
    deep_search: bool = Field(
        default=False,
        description="Default search mode: False for fast (metadata only), True for deep (full content).",
    )
    include_answer: bool = Field(
        default=False,
        description="Default setting for including LLM-generated answers.",
    )
    parsing_type: Literal["plain_text", "markdown", "simplified_html"] = Field(
        default="markdown",
        description="Content format. Markdown is recommended for LLM consumption.",
    )
    locale: str | None = Field(
        default=None, description="Locale for search results (e.g., 'en-US')."
    )
    country: str | None = Field(
        default=None,
        description="Country code for geo-targeted search (e.g., 'US').",
    )
    max_content_length_per_result: int = Field(
        default=1000,
        description="Maximum length for the 'content' of each search result to avoid context window issues.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["nimble-python"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="NIMBLE_API_KEY",
                description="API key for Nimble search service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        try:
            from nimble_python import AsyncNimble, Nimble
        except ImportError:
            try:
                import subprocess

                import click
            except ImportError as e:
                raise ImportError(
                    "The 'nimble-python' package is required. "
                    "Please install it with: uv add nimble-python"
                ) from e

            if click.confirm(
                "You are missing the 'nimble-python' package, which is required for NimbleSearchTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "nimble-python"], check=True)  # noqa: S607
                    from nimble_python import AsyncNimble, Nimble
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'nimble-python' but failed: {e}. "
                        f"Please install it manually to use the NimbleSearchTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'nimble-python' package is required to use the NimbleSearchTool. "
                    "Please install it with: uv add nimble-python"
                )

        headers = {
            "X-Client-Source": "crewai-tools",
            "X-Client-Tool": "NimbleSearchTool",
        }
        self.client = Nimble(api_key=self.api_key, default_headers=headers)
        self.async_client = AsyncNimble(
            api_key=self.api_key, default_headers=headers
        )

    def _convert_response(self, raw_results: Any) -> dict:
        """Convert SearchResponse object to dict."""
        return raw_results.model_dump()

    def _truncate_content(self, results_dict: dict) -> None:
        """Truncate content in results to max_content_length_per_result."""
        if (
            isinstance(results_dict, dict)
            and "results" in results_dict
            and isinstance(results_dict["results"], list)
        ):
            for item in results_dict["results"]:
                if (
                    isinstance(item, dict)
                    and "content" in item
                    and isinstance(item["content"], str)
                    and len(item["content"]) > self.max_content_length_per_result
                ):
                    item["content"] = (
                        item["content"][: self.max_content_length_per_result] + "..."
                    )

    def _build_search_params(
        self,
        query: str,
        max_results: int | None = None,
        deep_search: bool | None = None,
        include_answer: bool | None = None,
        time_range: Literal["hour", "day", "week", "month", "year"] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict:
        """Build search parameters from runtime overrides and init defaults."""
        search_params = {
            "query": query,
            "num_results": max_results if max_results is not None else self.max_results,
            "deep_search": deep_search if deep_search is not None else self.deep_search,
            "include_answer": (
                include_answer if include_answer is not None else self.include_answer
            ),
            "parsing_type": self.parsing_type,
        }

        if time_range is not None:
            search_params["time_range"] = time_range
        if include_domains is not None:
            search_params["include_domains"] = include_domains
        if exclude_domains is not None:
            search_params["exclude_domains"] = exclude_domains
        if self.locale is not None:
            search_params["locale"] = self.locale
        if self.country is not None:
            search_params["country"] = self.country

        return search_params

    def _process_response(self, raw_results: Any) -> str:
        """Convert raw API response to truncated JSON string."""
        results_dict = self._convert_response(raw_results)
        self._truncate_content(results_dict)
        return json.dumps(results_dict, indent=2)

    def _run(
        self,
        query: str,
        max_results: int | None = None,
        deep_search: bool | None = None,
        include_answer: bool | None = None,
        time_range: Literal["hour", "day", "week", "month", "year"] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> str:
        """Synchronously performs a search using the Nimble API."""
        if not self.client:
            raise ValueError(
                "Nimble client is not initialized. Ensure 'nimble-python' is installed and API key is set."
            )

        search_params = self._build_search_params(
            query, max_results, deep_search, include_answer,
            time_range, include_domains, exclude_domains,
        )
        raw_results = self.client.search(**search_params)
        return self._process_response(raw_results)

    async def _arun(
        self,
        query: str,
        max_results: int | None = None,
        deep_search: bool | None = None,
        include_answer: bool | None = None,
        time_range: Literal["hour", "day", "week", "month", "year"] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> str:
        """Asynchronously performs a search using the Nimble API."""
        if not self.async_client:
            raise ValueError(
                "Nimble async client is not initialized. Ensure 'nimble-python' is installed and API key is set."
            )

        search_params = self._build_search_params(
            query, max_results, deep_search, include_answer,
            time_range, include_domains, exclude_domains,
        )
        raw_results = await self.async_client.search(**search_params)
        return self._process_response(raw_results)
