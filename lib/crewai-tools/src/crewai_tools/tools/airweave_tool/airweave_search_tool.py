"""Airweave Search Tool for CrewAI.

Search across connected data sources (Stripe, GitHub, Notion, Slack, etc.)
using Airweave's unified search API.
"""

import os
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

# Module-level import with availability flag
try:
    from airweave import AirweaveSDK, AsyncAirweaveSDK
    AIRWEAVE_AVAILABLE = True
except ImportError:
    AIRWEAVE_AVAILABLE = False
    AirweaveSDK = Any  # type: ignore
    AsyncAirweaveSDK = Any  # type: ignore


class AirweaveSearchToolSchema(BaseModel):
    """Input schema for AirweaveSearchTool."""

    query: str = Field(
        ...,
        description="The search query to find relevant information from your connected data sources"
    )
    limit: Optional[int] = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return (1-100)"
    )
    offset: Optional[int] = Field(
        default=0,
        ge=0,
        description="Number of results to skip for pagination"
    )
    response_type: Optional[str] = Field(
        default="raw",
        description="Response format: 'raw' for search results or 'completion' for AI-generated answer"
    )
    recency_bias: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight for recent results (0.0=no bias, 1.0=only recency)"
    )


class AirweaveSearchTool(BaseTool):
    """
    Search across all connected data sources in an Airweave collection.
    
    This tool enables agents to search through any data source connected to Airweave,
    including Stripe, GitHub, Notion, Slack, HubSpot, Zendesk, and 50+ other integrations.
    
    Mirrors the client.collections.search() method from the Airweave Python SDK.
    Use this for straightforward searches. For advanced filtering and reranking,
    use AirweaveAdvancedSearchTool.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "Airweave Search"
    description: str = (
        "Search across all connected data sources in your Airweave collection. "
        "Use this to find information from Stripe, GitHub, Notion, Slack, and other integrated apps. "
        "Supports both raw search results and AI-generated answers via response_type parameter."
    )
    args_schema: Type[BaseModel] = AirweaveSearchToolSchema

    # Required configuration
    collection_id: str = Field(
        ...,
        description="The readable ID of the Airweave collection to search"
    )

    # Optional configuration
    base_url: Optional[str] = Field(
        default=None,
        description="Custom Airweave API base URL (for self-hosted instances)"
    )
    max_content_length: int = Field(
        default=300,
        description="Maximum content length to display per result"
    )

    # Dependencies
    package_dependencies: List[str] = ["airweave-sdk"]
    env_vars: List[EnvVar] = [
        EnvVar(
            name="AIRWEAVE_API_KEY",
            description="API key for Airweave (get from https://app.airweave.ai)",
            required=True
        ),
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Airweave search tool."""
        super().__init__(**kwargs)

        # Check if package is available, offer interactive installation
        if not AIRWEAVE_AVAILABLE:
            import click
            
            if click.confirm(
                "You are missing the 'airweave-sdk' package. Would you like to install it?"
            ):
                import subprocess
                try:
                    subprocess.run(["uv", "add", "airweave-sdk"], check=True)  # noqa: S607
                    # Import after installation
                    from airweave import AirweaveSDK, AsyncAirweaveSDK
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install airweave-sdk package") from e
            else:
                raise ImportError(
                    "`airweave-sdk` package not found, please run `uv add airweave-sdk`"
                ) from None

        # Validate API key
        api_key = os.getenv("AIRWEAVE_API_KEY")
        if not api_key:
            raise ValueError(
                "AIRWEAVE_API_KEY environment variable is required. "
                "Get your API key from https://app.airweave.ai"
            )

        # Get version safely (only once)
        try:
            from importlib.metadata import version
            package_version = version("crewai-tools")
        except Exception:
            package_version = "unknown"

        # Initialize client kwargs
        client_kwargs = {
            "api_key": api_key,
            "framework_name": "crewai",
            "framework_version": package_version,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        # Initialize both sync and async clients
        # Import from module namespace to avoid scope issues
        import crewai_tools.tools.airweave_tool.airweave_search_tool as airweave_module
        self._client = airweave_module.AirweaveSDK(**client_kwargs)
        self._async_client = airweave_module.AsyncAirweaveSDK(**client_kwargs)

    def _run(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        response_type: str = "raw",
        recency_bias: float = 0.0,
        **kwargs: Any
    ) -> str:
        """
        Execute search and return results.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination
            response_type: 'raw' for search results or 'completion' for AI answer
            recency_bias: Weight for recent results (0.0-1.0)
            
        Returns:
            Formatted string containing search results or AI-generated answer
        """
        try:
            # Validate response_type
            if response_type not in ["raw", "completion"]:
                response_type = "raw"

            response = self._client.collections.search(
                readable_id=self.collection_id,
                query=query,
                limit=limit,
                offset=offset,
                response_type=response_type,
                recency_bias=recency_bias
            )

            # Handle completion response
            if response_type == "completion":
                if response.completion:
                    return response.completion
                else:
                    return "Unable to generate an answer from available data. Try rephrasing your question."

            # Handle raw results response
            if response.status == "no_results":
                return "No results found for your query."

            if response.status == "no_relevant_results":
                return "Search completed but no sufficiently relevant results were found. Try rephrasing your query."

            # Format and return results
            return self._format_results(response.results, limit)

        except Exception as e:
            return f"Error performing search: {str(e)}"

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        response_type: str = "raw",
        recency_bias: float = 0.0,
        **kwargs: Any
    ) -> str:
        """
        Async implementation using AsyncAirweaveSDK.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination
            response_type: 'raw' for search results or 'completion' for AI answer
            recency_bias: Weight for recent results (0.0-1.0)
            
        Returns:
            Formatted string containing search results or AI-generated answer
        """
        try:
            # Validate response_type
            if response_type not in ["raw", "completion"]:
                response_type = "raw"

            response = await self._async_client.collections.search(
                readable_id=self.collection_id,
                query=query,
                limit=limit,
                offset=offset,
                response_type=response_type,
                recency_bias=recency_bias
            )

            # Handle completion response
            if response_type == "completion":
                if response.completion:
                    return response.completion
                else:
                    return "Unable to generate an answer from available data. Try rephrasing your question."

            # Handle raw results response
            if response.status == "no_results":
                return "No results found for your query."

            if response.status == "no_relevant_results":
                return "Search completed but no sufficiently relevant results were found. Try rephrasing your query."

            return self._format_results(response.results, limit)

        except Exception as e:
            return f"Error performing async search: {str(e)}"

    def _format_results(self, results: List[dict], limit: int) -> str:
        """
        Format search results for agent consumption.
        
        Args:
            results: List of search result dictionaries
            limit: Maximum number of results to format
            
        Returns:
            Human-readable formatted string
        """
        if not results:
            return "No results found."

        formatted = [f"Found {len(results)} result(s):\n"]

        for idx, result in enumerate(results[:limit], 1):
            payload = result.get("payload", {})
            score = result.get("score", 0.0)

            formatted.append(f"\n--- Result {idx} (Score: {score:.3f}) ---")

            # Content (truncate if too long)
            content = payload.get("md_content", "")
            if content:
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "..."
                formatted.append(f"Content: {content}")

            # Metadata
            if "source_name" in payload:
                formatted.append(f"Source: {payload['source_name']}")

            if "entity_id" in payload:
                formatted.append(f"Entity ID: {payload['entity_id']}")

            if "created_at" in payload:
                formatted.append(f"Created: {payload['created_at']}")

            if "url" in payload:
                formatted.append(f"URL: {payload['url']}")

        return "\n".join(formatted)
