"""ChromaDB client implementation."""

import logging
from typing import Any

from chromadb.api.types import (
    EmbeddingFunction as ChromaEmbeddingFunction,
    QueryResult,
)
from typing_extensions import Unpack

from crewai.rag.chromadb.types import (
    ChromaDBClientType,
    ChromaDBCollectionCreateParams,
    ChromaDBCollectionSearchParams,
)
from crewai.rag.chromadb.utils import (
    _create_batch_slice,
    _extract_search_params,
    _is_async_client,
    _is_sync_client,
    _prepare_documents_for_chromadb,
    _process_query_results,
    _sanitize_collection_name,
)
from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionAddParams,
    BaseCollectionParams,
)
from crewai.rag.types import SearchResult
from crewai.utilities.logger_utils import suppress_logging


class ChromaDBClient(BaseClient):
    """ChromaDB implementation of the BaseClient protocol.

    Provides vector database operations for ChromaDB, supporting both
    synchronous and asynchronous clients.

    Attributes:
        client: ChromaDB client instance (ClientAPI or AsyncClientAPI).
        embedding_function: Function to generate embeddings for documents.
        default_limit: Default number of results to return in searches.
        default_score_threshold: Default minimum score for search results.
    """

    def __init__(
        self,
        client: ChromaDBClientType,
        embedding_function: ChromaEmbeddingFunction,
        default_limit: int = 5,
        default_score_threshold: float = 0.6,
        default_batch_size: int = 100,
    ) -> None:
        """Initialize ChromaDBClient with client and embedding function.

        Args:
            client: Pre-configured ChromaDB client instance.
            embedding_function: Embedding function for text to vector conversion.
            default_limit: Default number of results to return in searches.
            default_score_threshold: Default minimum score for search results.
            default_batch_size: Default batch size for adding documents.
        """
        self.client = client
        self.embedding_function = embedding_function
        self.default_limit = default_limit
        self.default_score_threshold = default_score_threshold
        self.default_batch_size = default_batch_size

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
            ...     get_or_create=True,
            ... )
        """
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method create_collection() requires a ClientAPI. "
                "Use acreate_collection() for AsyncClientAPI."
            )

        metadata = kwargs.get("metadata", {})
        if "hnsw:space" not in metadata:
            metadata["hnsw:space"] = "cosine"

        self.client.create_collection(
            name=_sanitize_collection_name(kwargs["collection_name"]),
            configuration=kwargs.get("configuration"),  # type: ignore[arg-type]
            metadata=metadata,
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
            ...         get_or_create=True,
            ...     )
            >>> asyncio.run(main())
        """
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method acreate_collection() requires an AsyncClientAPI. "
                "Use create_collection() for ClientAPI."
            )

        metadata = kwargs.get("metadata", {})
        if "hnsw:space" not in metadata:
            metadata["hnsw:space"] = "cosine"

        await self.client.create_collection(
            name=_sanitize_collection_name(kwargs["collection_name"]),
            configuration=kwargs.get("configuration"),  # type: ignore[arg-type]
            metadata=metadata,
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
            ...     metadata={"description": "Product documentation"},
            ... )
        """
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method get_or_create_collection() requires a ClientAPI. "
                "Use aget_or_create_collection() for AsyncClientAPI."
            )

        metadata = kwargs.get("metadata", {})
        if "hnsw:space" not in metadata:
            metadata["hnsw:space"] = "cosine"

        return self.client.get_or_create_collection(
            name=_sanitize_collection_name(kwargs["collection_name"]),
            configuration=kwargs.get("configuration"),  # type: ignore[arg-type]
            metadata=metadata,
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
            ...         metadata={"description": "Product documentation"},
            ...     )
            >>> asyncio.run(main())
        """
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method aget_or_create_collection() requires an AsyncClientAPI. "
                "Use get_or_create_collection() for ClientAPI."
            )

        metadata = kwargs.get("metadata", {})
        if "hnsw:space" not in metadata:
            metadata["hnsw:space"] = "cosine"

        return await self.client.get_or_create_collection(
            name=_sanitize_collection_name(kwargs["collection_name"]),
            configuration=kwargs.get("configuration") or None,  # type: ignore[arg-type]
            metadata=metadata,
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
            batch_size: Optional batch size for processing documents (default: 100)

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
        batch_size = kwargs.get("batch_size", self.default_batch_size)

        if not documents:
            raise ValueError("Documents list cannot be empty")

        collection = self.client.get_or_create_collection(
            name=_sanitize_collection_name(collection_name),
            embedding_function=self.embedding_function,
        )

        prepared = _prepare_documents_for_chromadb(documents)

        for i in range(0, len(prepared.ids), batch_size):
            batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
                prepared=prepared, start_index=i, batch_size=batch_size
            )

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,  # type: ignore[arg-type]
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
            batch_size: Optional batch size for processing documents (default: 100)

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
        batch_size = kwargs.get("batch_size", self.default_batch_size)

        if not documents:
            raise ValueError("Documents list cannot be empty")

        collection = await self.client.get_or_create_collection(
            name=_sanitize_collection_name(collection_name),
            embedding_function=self.embedding_function,
        )
        prepared = _prepare_documents_for_chromadb(documents)

        for i in range(0, len(prepared.ids), batch_size):
            batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
                prepared=prepared, start_index=i, batch_size=batch_size
            )

            await collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,  # type: ignore[arg-type]
            )

    def search(
        self, **kwargs: Unpack[ChromaDBCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query.

        Performs semantic search to find documents similar to the query text.
        Uses the configured embedding function to generate query embeddings.

        Keyword Args:
            collection_name: Name of the collection to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default: 10).
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1) for results.
            where: Optional ChromaDB where clause for metadata filtering.
            where_document: Optional ChromaDB where clause for document content filtering.
            include: Optional list of fields to include in results.

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            TypeError: If AsyncClientAPI is used instead of ClientAPI for sync operations.
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to ChromaDB server.
        """
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method search() requires a ClientAPI. "
                "Use asearch() for AsyncClientAPI."
            )

        if "limit" not in kwargs:
            kwargs["limit"] = self.default_limit
        if "score_threshold" not in kwargs:
            kwargs["score_threshold"] = self.default_score_threshold

        params = _extract_search_params(kwargs)

        collection = self.client.get_or_create_collection(
            name=_sanitize_collection_name(params.collection_name),
            embedding_function=self.embedding_function,
        )

        where = params.where if params.where is not None else params.metadata_filter

        with suppress_logging(
            "chromadb.segment.impl.vector.local_persistent_hnsw", logging.ERROR
        ):
            results: QueryResult = collection.query(
                query_texts=[params.query],
                n_results=params.limit,
                where=where,
                where_document=params.where_document,
                include=params.include,
            )

        return _process_query_results(
            collection=collection,
            results=results,
            params=params,
        )

    async def asearch(
        self, **kwargs: Unpack[ChromaDBCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously.

        Performs semantic search to find documents similar to the query text.
        Uses the configured embedding function to generate query embeddings.

        Keyword Args:
            collection_name: Name of the collection to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default: 10).
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1) for results.
            where: Optional ChromaDB where clause for metadata filtering.
            where_document: Optional ChromaDB where clause for document content filtering.
            include: Optional list of fields to include in results.

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            TypeError: If ClientAPI is used instead of AsyncClientAPI for async operations.
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to ChromaDB server.
        """
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method asearch() requires an AsyncClientAPI. "
                "Use search() for ClientAPI."
            )

        if "limit" not in kwargs:
            kwargs["limit"] = self.default_limit
        if "score_threshold" not in kwargs:
            kwargs["score_threshold"] = self.default_score_threshold

        params = _extract_search_params(kwargs)

        collection = await self.client.get_or_create_collection(
            name=_sanitize_collection_name(params.collection_name),
            embedding_function=self.embedding_function,
        )

        where = params.where if params.where is not None else params.metadata_filter

        with suppress_logging(
            "chromadb.segment.impl.vector.local_persistent_hnsw", logging.ERROR
        ):
            results: QueryResult = await collection.query(
                query_texts=[params.query],
                n_results=params.limit,
                where=where,
                where_document=params.where_document,
                include=params.include,
            )

        return _process_query_results(
            collection=collection,
            results=results,
            params=params,
        )

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data.

        Permanently removes a collection and all documents, embeddings, and metadata it contains.
        This operation cannot be undone.

        Keyword Args:
            collection_name: Name of the collection to delete.

        Raises:
            TypeError: If AsyncClientAPI is used instead of ClientAPI for sync operations.
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> client = ChromaDBClient()
            >>> client.delete_collection(collection_name="old_documents")
        """
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method delete_collection() requires a ClientAPI. "
                "Use adelete_collection() for AsyncClientAPI."
            )

        collection_name = kwargs["collection_name"]
        self.client.delete_collection(name=_sanitize_collection_name(collection_name))

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data asynchronously.

        Permanently removes a collection and all documents, embeddings, and metadata it contains.
        This operation cannot be undone.

        Keyword Args:
            collection_name: Name of the collection to delete.

        Raises:
            TypeError: If ClientAPI is used instead of AsyncClientAPI for async operations.
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> import asyncio
            >>> async def main():
            ...     client = ChromaDBClient()
            ...     await client.adelete_collection(collection_name="old_documents")
            >>> asyncio.run(main())
        """
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method adelete_collection() requires an AsyncClientAPI. "
                "Use delete_collection() for ClientAPI."
            )

        collection_name = kwargs["collection_name"]
        await self.client.delete_collection(
            name=_sanitize_collection_name(collection_name)
        )

    def reset(self) -> None:
        """Reset the vector database by deleting all collections and data.

        Completely clears the ChromaDB instance, removing all collections,
        documents, embeddings, and metadata. This operation cannot be undone.
        Use with extreme caution in production environments.

        Raises:
            TypeError: If AsyncClientAPI is used instead of ClientAPI for sync operations.
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> client = ChromaDBClient()
            >>> client.reset()  # Removes ALL data from ChromaDB
        """
        if not _is_sync_client(self.client):
            raise TypeError(
                "Synchronous method reset() requires a ClientAPI. "
                "Use areset() for AsyncClientAPI."
            )

        self.client.reset()

    async def areset(self) -> None:
        """Reset the vector database by deleting all collections and data asynchronously.

        Completely clears the ChromaDB instance, removing all collections,
        documents, embeddings, and metadata. This operation cannot be undone.
        Use with extreme caution in production environments.

        Raises:
            TypeError: If ClientAPI is used instead of AsyncClientAPI for async operations.
            ConnectionError: If unable to connect to ChromaDB server.

        Example:
            >>> import asyncio
            >>> async def main():
            ...     client = ChromaDBClient()
            ...     await client.areset()  # Removes ALL data from ChromaDB
            >>> asyncio.run(main())
        """
        if not _is_async_client(self.client):
            raise TypeError(
                "Asynchronous method areset() requires an AsyncClientAPI. "
                "Use reset() for ClientAPI."
            )

        await self.client.reset()
