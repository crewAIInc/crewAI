"""ChromaDB client implementation."""

from typing import Any

from chromadb.api.configuration import CollectionConfigurationInterface
from chromadb.api.types import (
    CollectionMetadata,
    DataLoader,
    Embeddable,
    EmbeddingFunction as ChromaEmbeddingFunction,
    Loadable,
)
from typing_extensions import Unpack

from crewai.rag.chromadb.types import ChromaDBClientType
from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionParams,
    BaseCollectionAddParams,
    BaseCollectionSearchParams,
)
from crewai.rag.types import (
    SearchResult,
)


class ChromaDBCollectionCreateParams(BaseCollectionParams, total=False):
    """Parameters for creating a ChromaDB collection.

    This class extends BaseCollectionParams to include any additional
    parameters specific to ChromaDB collection creation.
    """

    configuration: CollectionConfigurationInterface
    metadata: CollectionMetadata
    embedding_function: ChromaEmbeddingFunction[Embeddable]
    data_loader: DataLoader[Loadable]
    get_or_create: bool


class ChromaDBClient(BaseClient):
    """ChromaDB implementation of the BaseClient protocol.

    Note: This is currently a stub implementation with all methods
    raising NotImplementedError. Full implementation coming soon.
    """

    client: ChromaDBClientType
    embedding_function: ChromaEmbeddingFunction[Embeddable]

    def create_collection(
        self, **kwargs: Unpack[ChromaDBCollectionCreateParams]
    ) -> None:
        """Create a new collection in ChromaDB.

        Uses the client's default embedding function if none provided.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.
            configuration: Optional collection configuration specifying distance metrics,
                HNSW parameters, or other backend-specific settings.
            metadata: Optional metadata dictionary to attach to the collection.
            embedding_function: Optional custom embedding function. If not provided,
                uses the client's default embedding function.
            data_loader: Optional data loader for batch loading data into the collection.
            get_or_create: If True, returns existing collection if it already exists
                instead of raising an error. Defaults to False.

        Raises:
            ValueError: If collection with the same name already exists and get_or_create
                is False.
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> client = ChromaDBClient()
            >>> client.create_collection(
            ...     collection_name="documents",
            ...     metadata={"description": "Product documentation"},
            ...     get_or_create=True
            ... )
        """
        self.client.create_collection(
            name=kwargs["collection_name"],
            configuration=kwargs.get("configuration"),
            metadata=kwargs.get("metadata"),
            embedding_function=kwargs.get(
                "embedding_function", self.embedding_function
            ),
            data_loader=kwargs.get("data_loader"),
            get_or_create=kwargs.get("get_or_create", False),
        )

    async def acreate_collection(
        self, **kwargs: Unpack[ChromaDBCollectionCreateParams]
    ) -> None:
        """Create a new collection in ChromaDB asynchronously.

        Creates a new collection with the specified name and optional configuration.
        If an embedding function is not provided, uses the client's default embedding function.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.
            configuration: Optional collection configuration specifying distance metrics,
                HNSW parameters, or other backend-specific settings.
            metadata: Optional metadata dictionary to attach to the collection.
            embedding_function: Optional custom embedding function. If not provided,
                uses the client's default embedding function.
            data_loader: Optional data loader for batch loading data into the collection.
            get_or_create: If True, returns existing collection if it already exists
                instead of raising an error. Defaults to False.

        Raises:
            ValueError: If collection with the same name already exists and get_or_create
                is False.
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> import asyncio
            >>> async def main():
            ...     client = ChromaDBClient()
            ...     await client.acreate_collection(
            ...         collection_name="documents",
            ...         metadata={"description": "Product documentation"},
            ...         get_or_create=True
            ...     )
            >>> asyncio.run(main())
        """
        await self.client.create_collection(
            name=kwargs["collection_name"],
            configuration=kwargs.get("configuration"),
            metadata=kwargs.get("metadata"),
            embedding_function=kwargs.get(
                "embedding_function", self.embedding_function
            ),
            data_loader=kwargs.get("data_loader"),
            get_or_create=kwargs.get("get_or_create", False),
        )

    def get_or_create_collection(
        self, **kwargs: Unpack[ChromaDBCollectionCreateParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist.

        Returns existing collection if found, otherwise creates a new one.

        Keyword Args:
            collection_name: Name of the collection to get or create.
            configuration: Optional collection configuration specifying distance metrics,
                HNSW parameters, or other backend-specific settings.
            metadata: Optional metadata dictionary to attach to the collection.
            embedding_function: Optional custom embedding function. If not provided,
                uses the client's default embedding function.
            data_loader: Optional data loader for batch loading data into the collection.

        Returns:
            A ChromaDB Collection object.

        Raises:
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> client = ChromaDBClient()
            >>> collection = client.get_or_create_collection(
            ...     collection_name="documents",
            ...     metadata={"description": "Product documentation"}
            ... )
        """
        return self.client.get_or_create_collection(
            name=kwargs["collection_name"],
            configuration=kwargs.get("configuration"),
            metadata=kwargs.get("metadata"),
            embedding_function=kwargs.get(
                "embedding_function", self.embedding_function
            ),
            data_loader=kwargs.get("data_loader"),
        )

    async def aget_or_create_collection(
        self, **kwargs: Unpack[ChromaDBCollectionCreateParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist asynchronously.

        Returns existing collection if found, otherwise creates a new one.

        Keyword Args:
            collection_name: Name of the collection to get or create.
            configuration: Optional collection configuration specifying distance metrics,
                HNSW parameters, or other backend-specific settings.
            metadata: Optional metadata dictionary to attach to the collection.
            embedding_function: Optional custom embedding function. If not provided,
                uses the client's default embedding function.
            data_loader: Optional data loader for batch loading data into the collection.

        Returns:
            A ChromaDB AsyncCollection object.

        Raises:
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> import asyncio
            >>> async def main():
            ...     client = ChromaDBClient()
            ...     collection = await client.aget_or_create_collection(
            ...         collection_name="documents",
            ...         metadata={"description": "Product documentation"}
            ...     )
            >>> asyncio.run(main())
        """
        return await self.client.get_or_create_collection(
            name=kwargs["collection_name"],
            configuration=kwargs.get("configuration"),
            metadata=kwargs.get("metadata"),
            embedding_function=kwargs.get(
                "embedding_function", self.embedding_function
            ),
            data_loader=kwargs.get("data_loader"),
        )

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
