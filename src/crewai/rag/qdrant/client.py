"""Qdrant client implementation."""

import asyncio
from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, QdrantClient as SyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from typing_extensions import Unpack

from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionParams,
    BaseCollectionAddParams,
    BaseCollectionSearchParams,
)
from crewai.rag.qdrant.types import QdrantClientType, QdrantCollectionCreateParams
from crewai.rag.qdrant.utils import prepare_search_params, process_search_results
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
    embedding_function: Any

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
        if not isinstance(self.client, SyncQdrantClient):
            raise TypeError(
                "Synchronous method create_collection() requires a QdrantClient. "
                "Use acreate_collection() for AsyncQdrantClient."
            )

        collection_name = kwargs["collection_name"]

        if self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=kwargs.get(
                "vectors_config", VectorParams(size=1536, distance=Distance.COSINE)
            ),
            sparse_vectors_config=kwargs.get("sparse_vectors_config"),
            shard_number=kwargs.get("shard_number"),
            sharding_method=kwargs.get("sharding_method"),
            replication_factor=kwargs.get("replication_factor"),
            write_consistency_factor=kwargs.get("write_consistency_factor"),
            on_disk_payload=kwargs.get("on_disk_payload"),
            hnsw_config=kwargs.get("hnsw_config"),
            optimizers_config=kwargs.get("optimizers_config"),
            wal_config=kwargs.get("wal_config"),
            quantization_config=kwargs.get("quantization_config"),
            init_from=kwargs.get("init_from"),
            timeout=kwargs.get("timeout"),
        )

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
        if not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "Asynchronous method acreate_collection() requires an AsyncQdrantClient. "
                "Use create_collection() for QdrantClient."
            )

        collection_name = kwargs["collection_name"]

        if await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=kwargs.get(
                "vectors_config", VectorParams(size=1536, distance=Distance.COSINE)
            ),
            sparse_vectors_config=kwargs.get("sparse_vectors_config"),
            shard_number=kwargs.get("shard_number"),
            sharding_method=kwargs.get("sharding_method"),
            replication_factor=kwargs.get("replication_factor"),
            write_consistency_factor=kwargs.get("write_consistency_factor"),
            on_disk_payload=kwargs.get("on_disk_payload"),
            hnsw_config=kwargs.get("hnsw_config"),
            optimizers_config=kwargs.get("optimizers_config"),
            wal_config=kwargs.get("wal_config"),
            quantization_config=kwargs.get("quantization_config"),
            init_from=kwargs.get("init_from"),
            timeout=kwargs.get("timeout"),
        )

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
        if not isinstance(self.client, SyncQdrantClient):
            raise TypeError(
                "Synchronous method get_or_create_collection() requires a QdrantClient. "
                "Use aget_or_create_collection() for AsyncQdrantClient."
            )

        collection_name = kwargs["collection_name"]

        if self.client.collection_exists(collection_name):
            return self.client.get_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=kwargs.get(
                "vectors_config", VectorParams(size=1536, distance=Distance.COSINE)
            ),
            sparse_vectors_config=kwargs.get("sparse_vectors_config"),
            shard_number=kwargs.get("shard_number"),
            sharding_method=kwargs.get("sharding_method"),
            replication_factor=kwargs.get("replication_factor"),
            write_consistency_factor=kwargs.get("write_consistency_factor"),
            on_disk_payload=kwargs.get("on_disk_payload"),
            hnsw_config=kwargs.get("hnsw_config"),
            optimizers_config=kwargs.get("optimizers_config"),
            wal_config=kwargs.get("wal_config"),
            quantization_config=kwargs.get("quantization_config"),
            init_from=kwargs.get("init_from"),
            timeout=kwargs.get("timeout"),
        )

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
        if not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "Asynchronous method aget_or_create_collection() requires an AsyncQdrantClient. "
                "Use get_or_create_collection() for QdrantClient."
            )

        collection_name = kwargs["collection_name"]

        if await self.client.collection_exists(collection_name):
            return await self.client.get_collection(collection_name)

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=kwargs.get(
                "vectors_config", VectorParams(size=1536, distance=Distance.COSINE)
            ),
            sparse_vectors_config=kwargs.get("sparse_vectors_config"),
            shard_number=kwargs.get("shard_number"),
            sharding_method=kwargs.get("sharding_method"),
            replication_factor=kwargs.get("replication_factor"),
            write_consistency_factor=kwargs.get("write_consistency_factor"),
            on_disk_payload=kwargs.get("on_disk_payload"),
            hnsw_config=kwargs.get("hnsw_config"),
            optimizers_config=kwargs.get("optimizers_config"),
            wal_config=kwargs.get("wal_config"),
            quantization_config=kwargs.get("quantization_config"),
            init_from=kwargs.get("init_from"),
            timeout=kwargs.get("timeout"),
        )

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
        if not isinstance(self.client, SyncQdrantClient):
            raise TypeError(
                "Synchronous method add_documents() requires a QdrantClient. "
                "Use aadd_documents() for AsyncQdrantClient."
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        points = []
        for doc in documents:
            doc_id = doc.get("doc_id", str(uuid4()))

            embedding = self.embedding_function(doc["content"])
            if not isinstance(embedding, list):
                embedding = embedding.tolist()

            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"content": doc["content"], **(doc.get("metadata", {}))},
            )
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
        if not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "Asynchronous method aadd_documents() requires an AsyncQdrantClient. "
                "Use add_documents() for QdrantClient."
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        points = []
        for doc in documents:
            doc_id = doc.get("doc_id", str(uuid4()))

            if hasattr(self.embedding_function, "__call__"):
                if asyncio.iscoroutinefunction(self.embedding_function):
                    embedding = await self.embedding_function(doc["content"])
                else:
                    embedding = self.embedding_function(doc["content"])
            else:
                embedding = self.embedding_function(doc["content"])

            if not isinstance(embedding, list):
                embedding = embedding.tolist()

            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"content": doc["content"], **(doc.get("metadata", {}))},
            )
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
        if not isinstance(self.client, SyncQdrantClient):
            raise TypeError(
                "Synchronous method search() requires a QdrantClient. "
                "Use asearch() for AsyncQdrantClient."
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", 10)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold")

        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        query_embedding = self.embedding_function(query)

        search_kwargs = prepare_search_params(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

        response = self.client.query_points(**search_kwargs)
        return process_search_results(response)

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
        if not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "Asynchronous method asearch() requires an AsyncQdrantClient. "
                "Use search() for QdrantClient."
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", 10)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold")

        if not await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        if hasattr(self.embedding_function, "__call__"):
            if asyncio.iscoroutinefunction(self.embedding_function):
                query_embedding = await self.embedding_function(query)
            else:
                query_embedding = self.embedding_function(query)
        else:
            query_embedding = self.embedding_function(query)

        search_kwargs = prepare_search_params(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

        response = await self.client.query_points(**search_kwargs)
        return process_search_results(response)

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data.

        Keyword Args:
            collection_name: Name of the collection to delete.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not isinstance(self.client, SyncQdrantClient):
            raise TypeError(
                "Synchronous method delete_collection() requires a QdrantClient. "
                "Use adelete_collection() for AsyncQdrantClient."
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
        if not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "Asynchronous method adelete_collection() requires an AsyncQdrantClient. "
                "Use delete_collection() for QdrantClient."
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
        if not isinstance(self.client, SyncQdrantClient):
            raise TypeError(
                "Synchronous method reset() requires a QdrantClient. "
                "Use areset() for AsyncQdrantClient."
            )

        collections_response = self.client.get_collections()

        for collection in collections_response.collections:
            self.client.delete_collection(collection_name=collection.name)

    async def areset(self) -> None:
        """Reset the vector database by deleting all collections and data asynchronously.

        Raises:
            ConnectionError: If unable to connect to Qdrant server.
        """
        if not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "Asynchronous method areset() requires an AsyncQdrantClient. "
                "Use reset() for QdrantClient."
            )

        collections_response = await self.client.get_collections()

        for collection in collections_response.collections:
            await self.client.delete_collection(collection_name=collection.name)
