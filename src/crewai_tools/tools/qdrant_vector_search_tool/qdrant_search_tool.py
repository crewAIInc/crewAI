import json
import os
from typing import Any, Optional, Type


try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = Any  # type placeholder
    Filter = Any
    FieldCondition = Any
    MatchValue = Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class QdrantToolSchema(BaseModel):
    """Input for QdrantTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Qdrant database. Pass only the query, not the question.",
    )
    filter_by: Optional[str] = Field(
        default=None,
        description="Filter by properties. Pass only the properties, not the question.",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Filter by value. Pass only the value, not the question.",
    )


class QdrantVectorSearchTool(BaseTool):
    """Tool to query and filter results from a Qdrant database.

    This tool enables vector similarity search on internal documents stored in Qdrant,
    with optional filtering capabilities.

    Attributes:
        client: Configured QdrantClient instance
        collection_name: Name of the Qdrant collection to search
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score threshold
        qdrant_url: Qdrant server URL
        qdrant_api_key: Authentication key for Qdrant
    """

    model_config = {"arbitrary_types_allowed": True}
    client: QdrantClient = None
    name: str = "QdrantVectorSearchTool"
    description: str = "A tool to search the Qdrant database for relevant information on internal documents."
    args_schema: Type[BaseModel] = QdrantToolSchema
    query: Optional[str] = None
    filter_by: Optional[str] = None
    filter_value: Optional[str] = None
    collection_name: Optional[str] = None
    limit: Optional[int] = Field(default=3)
    score_threshold: float = Field(default=0.35)
    qdrant_url: str = Field(
        ...,
        description="The URL of the Qdrant server",
    )
    qdrant_api_key: str = Field(
        ...,
        description="The API key for the Qdrant server",
    )
    custom_embedding_fn: Optional[callable] = Field(
        default=None,
        description="A custom embedding function to use for vectorization. If not provided, the default model will be used.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
        else:
            import click

            if click.confirm(
                "The 'qdrant-client' package is required to use the QdrantVectorSearchTool. "
                "Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "qdrant-client"], check=True)
            else:
                raise ImportError(
                    "The 'qdrant-client' package is required to use the QdrantVectorSearchTool. "
                    "Please install it with: uv add qdrant-client"
                )

    def _run(
        self,
        query: str,
        filter_by: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> str:
        """Execute vector similarity search on Qdrant.

        Args:
            query: Search query to vectorize and match
            filter_by: Optional metadata field to filter on
            filter_value: Optional value to filter by

        Returns:
            JSON string containing search results with metadata and scores

        Raises:
            ImportError: If qdrant-client is not installed
            ValueError: If Qdrant credentials are missing
        """

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL is not set")

        # Create filter if filter parameters are provided
        search_filter = None
        if filter_by and filter_value:
            search_filter = Filter(
                must=[
                    FieldCondition(key=filter_by, match=MatchValue(value=filter_value))
                ]
            )

        # Search in Qdrant using the built-in query method
        query_vector = (
            self._vectorize_query(query)
            if not self.custom_embedding_fn
            else self.custom_embedding_fn(query)
        )
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=self.limit,
            score_threshold=self.score_threshold,
        )

        # Format results similar to storage implementation
        results = []
        # Extract the list of ScoredPoint objects from the tuple
        for point in search_results:
            result = {
                "metadata": point[1][0].payload.get("metadata", {}),
                "context": point[1][0].payload.get("text", ""),
                "distance": point[1][0].score,
            }
            results.append(result)

        return json.dumps(results, indent=2)

    def _vectorize_query(self, query: str) -> list[float]:
        """Default vectorization function with openai.

        Args:
            query (str): The query to vectorize

        Returns:
            list[float]: The vectorized query
        """
        import openai

        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        embedding = (
            client.embeddings.create(
                input=[query],
                model="text-embedding-3-small",
            )
            .data[0]
            .embedding
        )
        return embedding
