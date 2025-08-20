"""ChromaDB client implementation."""

import hashlib
from collections.abc import Mapping
from typing import Any, TypeGuard

from chromadb.api import AsyncClientAPI, ClientAPI
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
    BaseRecord,
    SearchResult,
)


def _is_sync_client(client: ChromaDBClientType) -> TypeGuard[ClientAPI]:
    """Type guard to check if the client is a synchronous ClientAPI.

    Args:
        client: The client to check.

    Returns:
        True if the client is a ClientAPI, False otherwise.
    """
    return isinstance(client, ClientAPI)


def _is_async_client(client: ChromaDBClientType) -> TypeGuard[AsyncClientAPI]:
    """Type guard to check if the client is an asynchronous AsyncClientAPI.

    Args:
        client: The client to check.

    Returns:
        True if the client is an AsyncClientAPI, False otherwise.
    """
    return isinstance(client, AsyncClientAPI)


def _prepare_documents_for_chromadb(
    documents: list[BaseRecord],
) -> tuple[list[str], list[str], list[Mapping[str, str | int | float | bool]]]:
    """Prepare documents for ChromaDB by extracting IDs, texts, and metadata.

    Args:
        documents: List of BaseRecord documents to prepare.

    Returns:
        Tuple of (ids, texts, metadatas) ready for ChromaDB.
    """
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[Mapping[str, str | int | float | bool]] = []

    for doc in documents:
        if "doc_id" in doc:
            ids.append(doc["doc_id"])
        else:
            content_hash = hashlib.sha256(doc["content"].encode()).hexdigest()[:16]
            ids.append(content_hash)

        texts.append(doc["content"])
        metadata = doc.get("metadata")
        if metadata:
            if isinstance(metadata, list):
                # ChromaDB accepts Mapping types for metadata
                metadatas.append(metadata[0] if metadata else {})
            else:
                # Metadata is already a Mapping
                metadatas.append(metadata)
        else:
            metadatas.append({})

    return ids, texts, metadatas


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
            TypeError: If AsyncClientAPI is used instead of ClientAPI for sync operations.
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
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method create_collection() requires a ClientAPI. "
                "Use acreate_collection() for AsyncClientAPI."
            )

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
            TypeError: If ClientAPI is used instead of AsyncClientAPI for async operations.
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
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method acreate_collection() requires an AsyncClientAPI. "
                "Use create_collection() for ClientAPI."
            )

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
            TypeError: If AsyncClientAPI is used instead of ClientAPI for sync operations.
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> client = ChromaDBClient()
            >>> collection = client.get_or_create_collection(
            ...     collection_name="documents",
            ...     metadata={"description": "Product documentation"}
            ... )
        """
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method get_or_create_collection() requires a ClientAPI. "
                "Use aget_or_create_collection() for AsyncClientAPI."
            )

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
            TypeError: If ClientAPI is used instead of AsyncClientAPI for async operations.
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
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method aget_or_create_collection() requires an AsyncClientAPI. "
                "Use get_or_create_collection() for ClientAPI."
            )

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
        """Add documents with their embeddings to a collection.

        Performs an upsert operation - documents with existing IDs are updated.
        Generates embeddings automatically using the configured embedding function.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing:
                - content: The text content (required)
                - doc_id: Optional unique identifier (auto-generated if missing)
                - metadata: Optional metadata dictionary

        Raises:
            TypeError: If AsyncClientAPI is used instead of ClientAPI for sync operations.
            ValueError: If collection doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to ChromaDB server.
        """
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method add_documents() requires a ClientAPI. "
                "Use aadd_documents() for AsyncClientAPI."
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        ids, texts, metadatas = _prepare_documents_for_chromadb(documents)
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection asynchronously.

        Performs an upsert operation - documents with existing IDs are updated.
        Generates embeddings automatically using the configured embedding function.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing:
                - content: The text content (required)
                - doc_id: Optional unique identifier (auto-generated if missing)
                - metadata: Optional metadata dictionary

        Raises:
            TypeError: If ClientAPI is used instead of AsyncClientAPI for async operations.
            ValueError: If collection doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to ChromaDB server.
        """
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method aadd_documents() requires an AsyncClientAPI. "
                "Use add_documents() for ClientAPI."
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        collection = await self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )
        ids, texts, metadatas = _prepare_documents_for_chromadb(documents)
        await collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

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
