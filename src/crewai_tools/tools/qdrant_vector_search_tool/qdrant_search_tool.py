import json
from typing import Any, Optional, Type

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = Any
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
    """Tool to query and filter results from a Qdrant vector database.

    This tool provides functionality to perform semantic search operations on documents
    stored in a Qdrant collection, with optional filtering capabilities.

    Attributes:
        name (str): Name of the tool
        description (str): Description of the tool's functionality
        client (QdrantClient): Qdrant client instance
        collection_name (str): Name of the Qdrant collection to search
        limit (int): Maximum number of results to return
        score_threshold (float): Minimum similarity score threshold
    """

    name: str = "QdrantVectorSearchTool"
    description: str = "A tool to search the Qdrant database for relevant information on internal documents."
    args_schema: Type[BaseModel] = QdrantToolSchema

    model_config = {"arbitrary_types_allowed": True}
    client: Optional[QdrantClient] = None
    collection_name: str = Field(
        ...,
        description="The name of the Qdrant collection to search",
    )
    limit: Optional[int] = Field(default=3)
    score_threshold: float = Field(default=0.35)
    qdrant_url: str = Field(
        ...,
        description="The URL of the Qdrant server",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Qdrant server",
    )
    vectorizer: Optional[str] = Field(
        default="BAAI/bge-base-en-v1.5",
        description="The vectorizer to use for the Qdrant server",
    )

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        qdrant_api_key: Optional[str] = None,
        vectorizer: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the QdrantVectorSearchTool.

        Args:
            qdrant_url: URL of the Qdrant server
            collection_name: Name of the collection to search
            qdrant_api_key: Optional API key for authentication
            vectorizer: Optional model name for text vectorization

        Raises:
            ImportError: If qdrant-client package is not installed
            ConnectionError: If unable to connect to Qdrant server
        """
        kwargs["qdrant_url"] = qdrant_url
        kwargs["collection_name"] = collection_name
        kwargs["qdrant_api_key"] = qdrant_api_key
        if vectorizer:
            kwargs["vectorizer"] = vectorizer

        super().__init__(**kwargs)

        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                )
                # Verify connection
                self.client.get_collections()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Qdrant server: {str(e)}")
        else:
            import click

            if click.confirm(
                "You are missing the 'qdrant-client' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(
                    ["uv", "add", "crewai[tools]", "qdrant-client"], check=True
                )
            else:
                raise ImportError(
                    "The 'qdrant-client' package is required to use the QdrantVectorSearchTool. "
                    "Please install it with: uv add crewai[tools] qdrant-client"
                )
        if vectorizer:
            self.client.set_model(self.vectorizer)

    def _run(
        self,
        query: str,
        filter_by: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> str:
        """Execute the vector search query.

        Args:
            query: Search query text
            filter_by: Optional field name to filter results
            filter_value: Optional value to filter by

        Returns:
            JSON string containing search results with metadata

        Raises:
            ValueError: If filter_by is provided without filter_value or vice versa
        """
        if bool(filter_by) != bool(filter_value):
            raise ValueError(
                "Both filter_by and filter_value must be provided together"
            )

        search_filter = None
        if filter_by and filter_value:
            search_filter = Filter(
                must=[
                    FieldCondition(key=filter_by, match=MatchValue(value=filter_value))
                ]
            )

        try:
            search_results = self.client.query(
                collection_name=self.collection_name,
                query_text=[query],
                query_filter=search_filter,
                limit=self.limit,
                score_threshold=self.score_threshold,
            )

            results = [
                {
                    "id": point.id,
                    "metadata": point.metadata,
                    "context": point.document,
                    "score": point.score,
                }
                for point in search_results
            ]

            if not results:
                return json.dumps({"message": "No results found", "results": []})

            return json.dumps(results, indent=2)

        except Exception as e:
            raise RuntimeError(f"Error executing Qdrant search: {str(e)}")
