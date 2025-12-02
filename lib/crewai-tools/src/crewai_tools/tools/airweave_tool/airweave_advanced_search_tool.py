"""Airweave Advanced Search Tool with filtering and reranking."""

import os
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class AirweaveAdvancedSearchToolSchema(BaseModel):
    """Input schema for AirweaveAdvancedSearchTool."""

    query: str = Field(
        ...,
        description="The search query to find relevant information"
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
    source_filter: Optional[str] = Field(
        default=None,
        description="Filter by specific source name (e.g., 'Stripe', 'GitHub', 'Slack')"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0-1.0)"
    )
    recency_bias: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for recent results (0.0=no bias, 1.0=only recency). Default: 0.3"
    )
    enable_reranking: Optional[bool] = Field(
        default=True,
        description="Enable AI reranking for better relevance"
    )
    search_method: Optional[str] = Field(
        default="hybrid",
        description="Search method: 'hybrid' (default), 'neural', or 'keyword'"
    )


class AirweaveAdvancedSearchTool(BaseTool):
    """
    Advanced search across Airweave collections with filtering and reranking.
    
    This tool provides advanced search capabilities including:
    - Source filtering (search only specific data sources)
    - Recency bias (prioritize recent results)
    - Score threshold filtering
    - AI-powered reranking for improved relevance
    - Query expansion for better recall
    - Multiple search methods (hybrid, neural, keyword)
    
    Mirrors the client.collections.search_advanced() method from the Airweave Python SDK.
    Use this when you need filtering, reranking, or fine-tuned search control.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "Airweave Advanced Search"
    description: str = (
        "Advanced search with filtering and AI enhancements. Use this when you need to: "
        "filter by specific sources, prioritize recent results, set minimum relevance scores, "
        "enable AI reranking, or use specific search methods (hybrid/neural/keyword)."
    )
    args_schema: Type[BaseModel] = AirweaveAdvancedSearchToolSchema

    # Required configuration
    collection_id: str = Field(
        ...,
        description="The readable ID of the Airweave collection to search"
    )

    # Optional configuration
    base_url: Optional[str] = Field(
        default=None,
        description="Custom Airweave API base URL"
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
            description="API key for Airweave",
            required=True
        ),
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the advanced search tool."""
        super().__init__(**kwargs)

        # Lazy import
        try:
            from airweave import AirweaveSDK
        except ImportError:
            raise ImportError(
                "Missing required package 'airweave-sdk'. Install with:\n"
                "  pip install airweave-sdk\n"
                "or\n"
                "  pip install 'crewai-tools[airweave]'"
            )

        # Validate API key
        api_key = os.getenv("AIRWEAVE_API_KEY")
        if not api_key:
            raise ValueError(
                "AIRWEAVE_API_KEY environment variable is required."
            )

        # Get version safely
        try:
            from importlib.metadata import version
            package_version = version("crewai-tools")
        except Exception:
            package_version = "unknown"

        # Initialize client
        client_kwargs = {
            "api_key": api_key,
            "framework_name": "crewai",
            "framework_version": package_version,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = AirweaveSDK(**client_kwargs)

    def _run(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        response_type: str = "raw",
        source_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
        recency_bias: float = 0.3,
        enable_reranking: bool = True,
        search_method: str = "hybrid",
        **kwargs: Any
    ) -> str:
        """Execute advanced search with filters."""
        try:
            # Validate response_type
            if response_type not in ["raw", "completion"]:
                response_type = "raw"

            # Validate search_method
            if search_method not in ["hybrid", "neural", "keyword"]:
                search_method = "hybrid"

            # Build filter if source_filter provided
            filter_obj = None
            if source_filter:
                from airweave import FieldCondition, Filter, MatchValue

                filter_obj = Filter(
                    must=[
                        FieldCondition(
                            key="source_name",
                            match=MatchValue(value=source_filter)
                        )
                    ]
                )

            # Perform advanced search
            response = self._client.collections.search_advanced(
                readable_id=self.collection_id,
                query=query,
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                recency_bias=recency_bias,
                enable_reranking=enable_reranking,
                search_method=search_method,
                filter=filter_obj,
                response_type=response_type
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
                return "Search completed but no sufficiently relevant results were found. Try adjusting filters or threshold."

            return self._format_results(response.results, limit, source_filter)

        except Exception as e:
            return f"Error performing advanced search: {str(e)}"

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        response_type: str = "raw",
        source_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
        recency_bias: float = 0.3,
        enable_reranking: bool = True,
        search_method: str = "hybrid",
        **kwargs: Any
    ) -> str:
        """Async implementation of advanced search."""
        # Initialize async client if needed
        if not hasattr(self, "_async_client"):
            from airweave import AsyncAirweaveSDK

            api_key = os.getenv("AIRWEAVE_API_KEY")
            
            # Get version safely
            try:
                from importlib.metadata import version
                package_version = version("crewai-tools")
            except Exception:
                package_version = "unknown"
            
            client_kwargs = {
                "api_key": api_key,
                "framework_name": "crewai",
                "framework_version": package_version,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._async_client = AsyncAirweaveSDK(**client_kwargs)

        try:
            # Validate response_type
            if response_type not in ["raw", "completion"]:
                response_type = "raw"

            # Validate search_method
            if search_method not in ["hybrid", "neural", "keyword"]:
                search_method = "hybrid"

            # Build filter
            filter_obj = None
            if source_filter:
                from airweave import FieldCondition, Filter, MatchValue

                filter_obj = Filter(
                    must=[
                        FieldCondition(
                            key="source_name",
                            match=MatchValue(value=source_filter)
                        )
                    ]
                )

            # Perform search
            response = await self._async_client.collections.search_advanced(
                readable_id=self.collection_id,
                query=query,
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                recency_bias=recency_bias,
                enable_reranking=enable_reranking,
                search_method=search_method,
                filter=filter_obj,
                response_type=response_type
            )

            # Handle completion response
            if response_type == "completion":
                if response.completion:
                    return response.completion
                else:
                    return "Unable to generate an answer from available data."

            # Handle raw results response
            if response.status == "no_results":
                return "No results found."

            if response.status == "no_relevant_results":
                return "Search completed but no sufficiently relevant results found."

            return self._format_results(response.results, limit, source_filter)

        except Exception as e:
            return f"Error in async advanced search: {str(e)}"

    def _format_results(
        self,
        results: List[dict],
        limit: int,
        source_filter: Optional[str] = None
    ) -> str:
        """Format advanced search results."""
        if not results:
            return "No results found."

        header = f"Found {len(results)} result(s)"
        if source_filter:
            header += f" from {source_filter}"
        header += ":\n"

        formatted = [header]

        for idx, result in enumerate(results[:limit], 1):
            payload = result.get("payload", {})
            score = result.get("score", 0.0)

            formatted.append(f"\n--- Result {idx} (Score: {score:.3f}) ---")

            # Content
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


