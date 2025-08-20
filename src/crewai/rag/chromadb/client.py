"""ChromaDB client implementation."""

from typing import Any
from typing_extensions import Unpack

from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionParams,
    BaseCollectionAddParams,
    BaseCollectionSearchParams,
)
from crewai.rag.types import (
    EmbeddingFunction,
    SearchResult,
)


class ChromaDBClient(BaseClient):
    """ChromaDB implementation of the BaseClient protocol.

    Note: This is currently a stub implementation with all methods
    raising NotImplementedError. Full implementation coming soon.
    """

    client: Any
    embedding_function: EmbeddingFunction

    def create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection/index in the vector database."""
        raise NotImplementedError

    async def acreate_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection/index in the vector database asynchronously."""
        raise NotImplementedError

    def get_or_create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> Any:
        """Get an existing collection or create it if it doesn't exist."""
        raise NotImplementedError

    async def aget_or_create_collection(
        self, **kwargs: Unpack[BaseCollectionParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist asynchronously."""
        raise NotImplementedError

    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection."""
        raise NotImplementedError

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection asynchronously."""
        raise NotImplementedError

    def search(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query."""
        raise NotImplementedError

    async def asearch(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously."""
        raise NotImplementedError

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data."""
        raise NotImplementedError

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data asynchronously."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the vector database by deleting all collections and data."""
        raise NotImplementedError

    async def areset(self) -> None:
        """Reset the vector database by deleting all collections and data asynchronously."""
        raise NotImplementedError
