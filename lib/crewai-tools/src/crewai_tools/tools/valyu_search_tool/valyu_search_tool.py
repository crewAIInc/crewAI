import asyncio
from collections.abc import Sequence
import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
try:
    from valyu import Valyu  # type: ignore[import-untyped]

    VALYU_AVAILABLE = True
except ImportError:
    VALYU_AVAILABLE = False
    Valyu = Any


class ValyuSearchToolSchema(BaseModel):
    """Input schema for ValyuSearchTool."""

    query: str = Field(..., description="The search query string.")


class ValyuSearchTool(BaseTool):
    """Tool that uses the Valyu Search API to perform unified searches across web, academic, financial, and proprietary data sources.

    Attributes:
        client: An instance of Valyu client.
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Valyu API key.
        search_type: The type of search to perform.
        max_num_results: The maximum number of results to return.
        relevance_threshold: Minimum relevance score for results.
        included_sources: A list of sources/domains to include in the search.
        excluded_sources: A list of sources/domains to exclude from the search.
        start_date: Start date for filtering results (YYYY-MM-DD).
        end_date: End date for filtering results (YYYY-MM-DD).
        response_length: Content length per result.
        country_code: 2-letter ISO country code to bias results geographically.
        max_content_length_per_result: Maximum length for the 'content' of each search result.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any | None = None
    name: str = "Valyu Search"
    description: str = (
        "A tool that performs unified searches across web, academic, financial, and proprietary data sources using the Valyu Search API. "
        "It returns a JSON object containing the search results."
    )
    args_schema: type[BaseModel] = ValyuSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("VALYU_API_KEY"),
        description="The Valyu API key. If not provided, it will be loaded from the environment variable VALYU_API_KEY.",
    )
    search_type: Literal["all", "web", "proprietary", "news"] = Field(
        default="all", description="The type of search to perform."
    )
    max_num_results: int = Field(
        default=10, description="The maximum number of results to return (1-20)."
    )
    relevance_threshold: float = Field(
        default=0.5,
        description="Minimum relevance score for results (0.0-1.0). Higher values return more relevant results.",
    )
    included_sources: Sequence[str] | None = Field(
        default=None,
        description="A list of specific sources or domains to include in the search.",
    )
    excluded_sources: Sequence[str] | None = Field(
        default=None,
        description="A list of specific sources or domains to exclude from the search.",
    )
    start_date: str | None = Field(
        default=None,
        description="Start date for filtering results (YYYY-MM-DD format).",
    )
    end_date: str | None = Field(
        default=None,
        description="End date for filtering results (YYYY-MM-DD format).",
    )
    response_length: Literal["short", "medium", "large", "max"] = Field(
        default="short", description="Content length per result."
    )
    country_code: str | None = Field(
        default=None,
        description="2-letter ISO country code to bias results geographically.",
    )
    max_content_length_per_result: int = Field(
        default=1000,
        description="Maximum length for the 'content' of each search result to avoid context window issues.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["valyu"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="VALYU_API_KEY",
                description="API key for Valyu search service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if VALYU_AVAILABLE:
            self.client = Valyu(api_key=self.api_key)
        else:
            try:
                import subprocess

                import click
            except ImportError as e:
                raise ImportError(
                    "The 'valyu' package is required. 'click' and 'subprocess' are also needed to assist with installation if the package is missing. "
                    "Please install 'valyu' manually (e.g., 'pip install valyu') and ensure 'click' and 'subprocess' are available."
                ) from e

            if click.confirm(
                "You are missing the 'valyu' package, which is required for ValyuSearchTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "valyu"], check=True)  # noqa: S607
                    raise ImportError(
                        "'valyu' has been installed. Please restart your Python application to use the ValyuSearchTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'valyu' but failed: {e}. "
                        f"Please install it manually to use the ValyuSearchTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'valyu' package is required to use the ValyuSearchTool. "
                    "Please install it with: pip install valyu"
                )

    def _run(
        self,
        query: str,
    ) -> str:
        """Synchronously performs a search using the Valyu API.
        Content of each result is truncated to `max_content_length_per_result`.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results with truncated content.
        """
        if not self.client:
            raise ValueError(
                "Valyu client is not initialized. Ensure 'valyu' is installed and API key is set."
            )

        # Build search parameters
        search_params: dict[str, Any] = {
            "query": query,
            "search_type": self.search_type,
            "max_num_results": self.max_num_results,
            "relevance_threshold": self.relevance_threshold,
            "response_length": self.response_length,
        }

        # Add optional parameters if set
        if self.included_sources:
            search_params["included_sources"] = list(self.included_sources)
        if self.excluded_sources:
            search_params["excluded_sources"] = list(self.excluded_sources)
        if self.start_date:
            search_params["start_date"] = self.start_date
        if self.end_date:
            search_params["end_date"] = self.end_date
        if self.country_code:
            search_params["country_code"] = self.country_code

        raw_results = self.client.search(**search_params)

        # Convert response to dict if it's a Pydantic model
        if hasattr(raw_results, "model_dump"):
            raw_results = raw_results.model_dump()
        elif hasattr(raw_results, "dict"):
            raw_results = raw_results.dict()

        # Truncate content if needed
        if (
            isinstance(raw_results, dict)
            and "results" in raw_results
            and isinstance(raw_results["results"], list)
        ):
            for item in raw_results["results"]:
                if (
                    isinstance(item, dict)
                    and "content" in item
                    and isinstance(item["content"], str)
                ):
                    if len(item["content"]) > self.max_content_length_per_result:
                        item["content"] = (
                            item["content"][: self.max_content_length_per_result]
                            + "..."
                        )

        return json.dumps(raw_results, indent=2)

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Asynchronously performs a search using the Valyu API.
        Content of each result is truncated to `max_content_length_per_result`.

        Note: The Valyu SDK currently uses synchronous calls, so this method
        runs the synchronous implementation in a thread pool to avoid blocking
        the event loop.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results with truncated content.
        """
        return await asyncio.to_thread(self._run, query)
