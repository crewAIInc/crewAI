"""Qdrant client implementation."""

from typing import Any, cast

from fastembed import TextEmbedding
from qdrant_client import QdrantClient as SyncQdrantClientBase
from typing_extensions import Unpack

from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionParams,
    BaseCollectionAddParams,
    BaseCollectionSearchParams,
)
from crewai.rag.core.exceptions import ClientMethodMismatchError
from crewai.rag.qdrant.types import (
    AsyncEmbeddingFunction,
    EmbeddingFunction,
    QdrantClientParams,
    QdrantClientType,
    QdrantCollectionCreateParams,
)
from crewai.rag.qdrant.utils import (
    _is_async_client,
    _is_async_embedding_function,
    _is_sync_client,
    _create_point_from_document,
    _get_collection_params,
    _prepare_search_params,
    _process_search_results,
)
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
    embedding_function: EmbeddingFunction | AsyncEmbeddingFunction

    def __init__(
        self,
        client: QdrantClientType | None = None,
        embedding_function: EmbeddingFunction | AsyncEmbeddingFunction | None = None,
        **kwargs: Unpack[QdrantClientParams],
    ) -> None:
        """Initialize QdrantClient with optional client and embedding function.

        Args:
            client: Optional pre-configured Qdrant client instance.
            embedding_function: Optional embedding function. If not provided,
                uses FastEmbed's BAAI/bge-small-en-v1.5 model.
            **kwargs: Additional arguments for QdrantClient creation.
        """
        if client is not None:
            self.client = client
        else:
            location = kwargs.get("location", ":memory:")
            client_kwargs = {k: v for k, v in kwargs.items() if k != "location"}
            self.client = SyncQdrantClientBase(location, **cast(Any, client_kwargs))

        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

            def _embed_fn(text: str) -> list[float]:
                embeddings = list(_embedder.embed([text]))
                return [float(x) for x in embeddings[0]] if embeddings else []

            self.embedding_function = _embed_fn

    def create_collection(self, **kwargs: Unpack[QdrantCollectionCreateParams]) -> None:
        """Create a new collection in Qdrant.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.
            vectors_config: Optional vector configuration. Defaults to 1536 dimensions with cosine distance.
            sparse_vectors_config: Optional sparse vector configuration.
            shard_number: Optional number of shards.
            replication_factor: Optional replication factor.
            write_consistency_factor: Optional write consistency factor.
            on_disk_payload: Optional flag to store payload on disk.
            hnsw_config: Optional HNSW index configuration.
            optimizers_config: Optional optimizer configuration.
            wal_config: Optional write-ahead log configuration.
            quantization_config: Optional quantization configuration.
            init_from: Optional collection to initialize from.
            timeout: Optional timeout for the operation.

        Raises:
            ValueError: If collection with the same name already exists.
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="create_collection",
                expected_client="QdrantClient",
                alt_method="acreate_collection",
                alt_client="AsyncQdrantClient",
            )

        collection_name = kwargs["collection_name"]

        if self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")

        params = _get_collection_params(kwargs)
        self.client.create_collection(**params)

    async def acreate_collection(
        self, **kwargs: Unpack[QdrantCollectionCreateParams]
    ) -> None:
        """Create a new collection in Qdrant asynchronously.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.
            vectors_config: Optional vector configuration. Defaults to 1536 dimensions with cosine distance.
            sparse_vectors_config: Optional sparse vector configuration.
            shard_number: Optional number of shards.
            replication_factor: Optional replication factor.
            write_consistency_factor: Optional write consistency factor.
            on_disk_payload: Optional flag to store payload on disk.
            hnsw_config: Optional HNSW index configuration.
            optimizers_config: Optional optimizer configuration.
            wal_config: Optional write-ahead log configuration.
            quantization_config: Optional quantization configuration.
            init_from: Optional collection to initialize from.
            timeout: Optional timeout for the operation.

        Raises:
            ValueError: If collection with the same name already exists.
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="acreate_collection",
                expected_client="AsyncQdrantClient",
                alt_method="create_collection",
                alt_client="QdrantClient",
            )

        collection_name = kwargs["collection_name"]

        if await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")

        params = _get_collection_params(kwargs)
        await self.client.create_collection(**params)

    def get_or_create_collection(
        self, **kwargs: Unpack[QdrantCollectionCreateParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist.

        Keyword Args:
            collection_name: Name of the collection to get or create.
            vectors_config: Optional vector configuration. Defaults to 1536 dimensions with cosine distance.
            sparse_vectors_config: Optional sparse vector configuration.
            shard_number: Optional number of shards.
            replication_factor: Optional replication factor.
            write_consistency_factor: Optional write consistency factor.
            on_disk_payload: Optional flag to store payload on disk.
            hnsw_config: Optional HNSW index configuration.
            optimizers_config: Optional optimizer configuration.
            wal_config: Optional write-ahead log configuration.
            quantization_config: Optional quantization configuration.
            init_from: Optional collection to initialize from.
            timeout: Optional timeout for the operation.

        Returns:
            Collection info dict with name and other metadata.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="get_or_create_collection",
                expected_client="QdrantClient",
                alt_method="aget_or_create_collection",
                alt_client="AsyncQdrantClient",
            )

        collection_name = kwargs["collection_name"]

        if self.client.collection_exists(collection_name):
            return self.client.get_collection(collection_name)

        params = _get_collection_params(kwargs)
        self.client.create_collection(**params)

        return self.client.get_collection(collection_name)

    async def aget_or_create_collection(
        self, **kwargs: Unpack[QdrantCollectionCreateParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist asynchronously.

        Keyword Args:
            collection_name: Name of the collection to get or create.
            vectors_config: Optional vector configuration. Defaults to 1536 dimensions with cosine distance.
            sparse_vectors_config: Optional sparse vector configuration.
            shard_number: Optional number of shards.
            replication_factor: Optional replication factor.
            write_consistency_factor: Optional write consistency factor.
            on_disk_payload: Optional flag to store payload on disk.
            hnsw_config: Optional HNSW index configuration.
            optimizers_config: Optional optimizer configuration.
            wal_config: Optional write-ahead log configuration.
            quantization_config: Optional quantization configuration.
            init_from: Optional collection to initialize from.
            timeout: Optional timeout for the operation.

        Returns:
            Collection info dict with name and other metadata.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="aget_or_create_collection",
                expected_client="AsyncQdrantClient",
                alt_method="get_or_create_collection",
                alt_client="QdrantClient",
            )

        collection_name = kwargs["collection_name"]

        if await self.client.collection_exists(collection_name):
            return await self.client.get_collection(collection_name)

        params = _get_collection_params(kwargs)
        await self.client.create_collection(**params)

        return await self.client.get_collection(collection_name)

    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing document data.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="add_documents",
                expected_client="QdrantClient",
                alt_method="aadd_documents",
                alt_client="AsyncQdrantClient",
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        points = []
        for doc in documents:
            if _is_async_embedding_function(self.embedding_function):
                raise TypeError(
                    "Async embedding function cannot be used with sync add_documents. "
                    "Use aadd_documents instead."
                )
            sync_fn = cast(EmbeddingFunction, self.embedding_function)
            embedding = sync_fn(doc["content"])
            point = _create_point_from_document(doc, embedding)
            points.append(point)

        self.client.upsert(collection_name=collection_name, points=points, wait=True)

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection asynchronously.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing document data.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="aadd_documents",
                expected_client="AsyncQdrantClient",
                alt_method="add_documents",
                alt_client="QdrantClient",
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        points = []
        for doc in documents:
            if _is_async_embedding_function(self.embedding_function):
                async_fn = cast(AsyncEmbeddingFunction, self.embedding_function)
                embedding = await async_fn(doc["content"])
            else:
                sync_fn = cast(EmbeddingFunction, self.embedding_function)
                embedding = sync_fn(doc["content"])
            point = _create_point_from_document(doc, embedding)
            points.append(point)

        await self.client.upsert(
            collection_name=collection_name, points=points, wait=True
        )

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
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="search",
                expected_client="QdrantClient",
                alt_method="asearch",
                alt_client="AsyncQdrantClient",
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", 10)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold")

        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        if _is_async_embedding_function(self.embedding_function):
            raise TypeError(
                "Async embedding function cannot be used with sync search. "
                "Use asearch instead."
            )
        sync_fn = cast(EmbeddingFunction, self.embedding_function)
        query_embedding = sync_fn(query)

        search_kwargs = _prepare_search_params(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

        response = self.client.query_points(**search_kwargs)
        return _process_search_results(response)

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
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="asearch",
                expected_client="AsyncQdrantClient",
                alt_method="search",
                alt_client="QdrantClient",
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", 10)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold")

        if not await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        if _is_async_embedding_function(self.embedding_function):
            async_fn = cast(AsyncEmbeddingFunction, self.embedding_function)
            query_embedding = await async_fn(query)
        else:
            sync_fn = cast(EmbeddingFunction, self.embedding_function)
            query_embedding = sync_fn(query)

        search_kwargs = _prepare_search_params(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

        response = await self.client.query_points(**search_kwargs)
        return _process_search_results(response)

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data.

        Keyword Args:
            collection_name: Name of the collection to delete.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="delete_collection",
                expected_client="QdrantClient",
                alt_method="adelete_collection",
                alt_client="AsyncQdrantClient",
            )

        collection_name = kwargs["collection_name"]

        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        self.client.delete_collection(collection_name=collection_name)

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data asynchronously.

        Keyword Args:
            collection_name: Name of the collection to delete.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="adelete_collection",
                expected_client="AsyncQdrantClient",
                alt_method="delete_collection",
                alt_client="QdrantClient",
            )

        collection_name = kwargs["collection_name"]

        if not await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        await self.client.delete_collection(collection_name=collection_name)

    def reset(self) -> None:
        """Reset the vector database by deleting all collections and data.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="reset",
                expected_client="QdrantClient",
                alt_method="areset",
                alt_client="AsyncQdrantClient",
            )

        collections_response = self.client.get_collections()

        for collection in collections_response.collections:
            self.client.delete_collection(collection_name=collection.name)

    async def areset(self) -> None:
        """Reset the vector database by deleting all collections and data asynchronously.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="areset",
                expected_client="AsyncQdrantClient",
                alt_method="reset",
                alt_client="QdrantClient",
            )

        collections_response = await self.client.get_collections()

        for collection in collections_response.collections:
            await self.client.delete_collection(collection_name=collection.name)
