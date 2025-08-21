"""Qdrant client implementation."""

from typing import Any

from typing_extensions import Unpack

from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionParams,
    BaseCollectionAddParams,
    BaseCollectionSearchParams,
)
from crewai.rag.qdrant.types import QdrantClientType
from crewai.rag.types import SearchResult


class QdrantClient(BaseClient):
    """Qdrant implementation of the BaseClient protocol.

    Provides vector database operations for Qdrant, supporting both
    synchronous and asynchronous clients.

    Attributes:
        client: Qdrant client instance (QdrantClient or AsyncQdrantClient).
        embedding_function: Function to generate embeddings for documents.
    """

    client: QdrantClientType
    embedding_function: Any  # EmbeddingFunction

    def create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection in Qdrant.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.

        Raises:
            ValueError: If collection with the same name already exists.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    async def acreate_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection in Qdrant asynchronously.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.

        Raises:
            ValueError: If collection with the same name already exists.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    def get_or_create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> Any:
        """Get an existing collection or create it if it doesn't exist.

        Keyword Args:
            collection_name: Name of the collection to get or create.

        Returns:
            A collection reference or metadata.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    async def aget_or_create_collection(
        self, **kwargs: Unpack[BaseCollectionParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist asynchronously.

        Keyword Args:
            collection_name: Name of the collection to get or create.

        Returns:
            A collection reference or metadata.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing document data.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection asynchronously.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing document data.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    def search(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query.

        Keyword Args:
            collection_name: Name of the collection to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default: 10).
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1) for results.

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    async def asearch(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously.

        Keyword Args:
            collection_name: Name of the collection to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default: 10).
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1) for results.

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data.

        Keyword Args:
            collection_name: Name of the collection to delete.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data asynchronously.

        Keyword Args:
            collection_name: Name of the collection to delete.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the vector database by deleting all collections and data.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError

    async def areset(self) -> None:
        """Reset the vector database by deleting all collections and data asynchronously.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        raise NotImplementedError
