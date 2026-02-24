import json
import logging
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

try:
    from airweave import AirweaveSDK, AsyncAirweaveSDK
    from airweave.types import SearchRequest

    AIRWEAVE_AVAILABLE = True
except ImportError:
    AIRWEAVE_AVAILABLE = False
    AirweaveSDK = Any
    AsyncAirweaveSDK = Any
    SearchRequest = Any


class AirweaveSearchToolSchema(BaseModel):
    """Input schema for AirweaveSearchTool."""

    query: str = Field(
        ..., description="The search query text to find relevant documents and data."
    )
    collection_id: str = Field(
        ...,
        description="The unique readable identifier of the Airweave collection to search (e.g., 'finance-data-2024').",
    )
    limit: int = Field(
        default=10,
        description="Maximum number of search results to return (default: 10).",
    )
    generate_answer: bool = Field(
        default=False,
        description="Whether to generate an AI-powered natural language answer based on search results (default: False).",
    )
    expand_query: bool = Field(
        default=False,
        description="Generate query variations to improve recall (default: False).",
    )
    rerank: bool = Field(
        default=True,
        description="Reorder top candidate results for improved relevance (default: True).",
    )
    temporal_relevance: float | None = Field(
        default=None,
        description="Weight recent content higher than older content; 0 = no recency effect, 1 = only recent items matter (default: None).",
    )


class AirweaveSearchTool(BaseTool):
    """Tool that uses the Airweave API to search across synced organizational data.

    Airweave makes any app searchable for your agent by syncing data from various sources
    with minimal configuration. This tool allows agents to perform natural language searches
    across your organization's data collections with advanced features like query expansion,
    reranking, and AI-powered answer generation.

    Attributes:
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Airweave API key (loaded from environment).
        base_url: Optional custom base URL for Airweave API.
        framework_name: Framework identifier for Airweave tracking.
        framework_version: Framework version for Airweave tracking.
        timeout: Request timeout in seconds (default: 60).
        max_content_length_per_result: Maximum length for content of each result (default: 1000).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: AirweaveSDK | None = None
    async_client: AsyncAirweaveSDK | None = None

    name: str = "Search Airweave Collection"
    description: str = (
        "Search across your organization's data using Airweave. "
        "Supports natural language queries, answer generation, query expansion, and advanced filtering. "
        "Returns relevant documents and optionally generates AI-powered answers based on the search results."
    )

    args_schema: type[BaseModel] = AirweaveSearchToolSchema

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("AIRWEAVE_API_KEY"),
        description="The Airweave API key. If not provided, it will be loaded from the environment variable AIRWEAVE_API_KEY.",
    )

    base_url: str | None = Field(
        default=None,
        description="Optional custom base URL for Airweave API.",
    )

    framework_name: str = Field(
        default="crewai",
        description="Framework identifier for Airweave tracking.",
    )

    framework_version: str | None = Field(
        default=None,
        description="Framework version for Airweave tracking.",
    )

    timeout: int = Field(
        default=60,
        description="The timeout for API requests in seconds.",
    )

    max_content_length_per_result: int = Field(
        default=1000,
        description="Maximum length for the 'content' field of each search result to avoid context window issues.",
    )

    package_dependencies: list[str] = Field(default_factory=lambda: ["airweave-sdk"])

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AIRWEAVE_API_KEY",
                description="API key for Airweave service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        """Initialize the AirweaveSearchTool.

        Ensures the airweave-sdk package is installed and initializes both
        synchronous and asynchronous clients.
        """
        super().__init__(**kwargs)

        if AIRWEAVE_AVAILABLE:
            # Initialize synchronous client
            client_kwargs = {
                "api_key": self.api_key,
                "framework_name": self.framework_name,
                "timeout": self.timeout,
            }

            if self.framework_version:
                client_kwargs["framework_version"] = self.framework_version

            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = AirweaveSDK(**client_kwargs)

            # Initialize asynchronous client
            self.async_client = AsyncAirweaveSDK(**client_kwargs)
        else:
            try:
                import subprocess

                import click
            except ImportError as e:
                raise ImportError(
                    "The 'airweave-sdk' package is required. 'click' and 'subprocess' are also needed to assist with installation if the package is missing. "
                    "Please install 'airweave-sdk' manually (e.g., 'pip install airweave-sdk') and ensure 'click' and 'subprocess' are available."
                ) from e

            if click.confirm(
                "You are missing the 'airweave-sdk' package, which is required for AirweaveSearchTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "airweave-sdk"], check=True)  # noqa: S607
                    raise ImportError(
                        "'airweave-sdk' has been installed. Please restart your Python application to use the AirweaveSearchTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'airweave-sdk' but failed: {e}. "
                        f"Please install it manually to use the AirweaveSearchTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'airweave-sdk' package is required to use the AirweaveSearchTool. "
                    "Please install it with: uv add airweave-sdk"
                )

    def _format_results(self, response: Any, query: str) -> str:
        """Format search results for LLM consumption.

        Args:
            response: The response from Airweave API
            query: The original search query

        Returns:
            A JSON-formatted string containing search results and metadata
        """
        # Extract results from response
        results = getattr(response, "results", [])
        completion = getattr(response, "completion", None)

        # Truncate content in each result if needed
        formatted_results = []
        for result in results:
            if isinstance(result, dict):
                formatted_result = result.copy()

                # Truncate content field if it exists and is too long
                if "content" in formatted_result and isinstance(
                    formatted_result["content"], str
                ):
                    if (
                        len(formatted_result["content"])
                        > self.max_content_length_per_result
                    ):
                        formatted_result["content"] = (
                            formatted_result["content"][
                                : self.max_content_length_per_result
                            ]
                            + "..."
                        )

                formatted_results.append(formatted_result)
            else:
                formatted_results.append(result)

        # Build response object
        output = {
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results,
        }

        # Add completion if available
        if completion:
            output["completion"] = completion

        return json.dumps(output, indent=2, default=str)

    def _run(
        self,
        query: str,
        collection_id: str,
        limit: int = 10,
        generate_answer: bool = False,
        expand_query: bool = False,
        rerank: bool = True,
        temporal_relevance: float | None = None,
    ) -> str:
        """Synchronously search an Airweave collection.

        Args:
            query: The search query text
            collection_id: The unique collection identifier
            limit: Maximum number of results to return
            generate_answer: Whether to generate an AI answer
            expand_query: Whether to expand the query for better recall
            rerank: Whether to rerank results for better relevance
            temporal_relevance: Weight for recent content (0-1)

        Returns:
            A JSON string containing the search results

        Raises:
            ValueError: If the client is not initialized properly
        """
        if not self.client:
            raise ValueError(
                "Airweave client is not initialized. Ensure 'airweave-sdk' is installed and API key is set."
            )

        try:
            # Build search request
            search_request_kwargs = {
                "query": query,
                "limit": limit,
                "generate_answer": generate_answer,
                "expand_query": expand_query,
                "rerank": rerank,
            }

            if temporal_relevance is not None:
                search_request_kwargs["temporal_relevance"] = temporal_relevance

            search_request = SearchRequest(**search_request_kwargs)

            # Execute search
            logger.info(
                f"Searching Airweave collection '{collection_id}' with query: {query}"
            )
            response = self.client.collections.search(
                readable_id=collection_id,
                request=search_request,
            )

            # Format and return results
            return self._format_results(response, query)

        except Exception as e:
            error_msg = f"Error searching Airweave collection '{collection_id}': {e!s}"
            logger.error(error_msg)
            return json.dumps(
                {
                    "error": error_msg,
                    "query": query,
                    "collection_id": collection_id,
                }
            )

    async def _arun(
        self,
        query: str,
        collection_id: str,
        limit: int = 10,
        generate_answer: bool = False,
        expand_query: bool = False,
        rerank: bool = True,
        temporal_relevance: float | None = None,
    ) -> str:
        """Asynchronously search an Airweave collection.

        Args:
            query: The search query text
            collection_id: The unique collection identifier
            limit: Maximum number of results to return
            generate_answer: Whether to generate an AI answer
            expand_query: Whether to expand the query for better recall
            rerank: Whether to rerank results for better relevance
            temporal_relevance: Weight for recent content (0-1)

        Returns:
            A JSON string containing the search results

        Raises:
            ValueError: If the async client is not initialized properly
        """
        if not self.async_client:
            raise ValueError(
                "Airweave async client is not initialized. Ensure 'airweave-sdk' is installed and API key is set."
            )

        try:
            # Build search request
            search_request_kwargs = {
                "query": query,
                "limit": limit,
                "generate_answer": generate_answer,
                "expand_query": expand_query,
                "rerank": rerank,
            }

            if temporal_relevance is not None:
                search_request_kwargs["temporal_relevance"] = temporal_relevance

            search_request = SearchRequest(**search_request_kwargs)

            # Execute search
            logger.info(
                f"Searching Airweave collection '{collection_id}' with query: {query}"
            )
            response = await self.async_client.collections.search(
                readable_id=collection_id,
                request=search_request,
            )

            # Format and return results
            return self._format_results(response, query)

        except Exception as e:
            error_msg = f"Error searching Airweave collection '{collection_id}': {e!s}"
            logger.error(error_msg)
            return json.dumps(
                {
                    "error": error_msg,
                    "query": query,
                    "collection_id": collection_id,
                }
            )
