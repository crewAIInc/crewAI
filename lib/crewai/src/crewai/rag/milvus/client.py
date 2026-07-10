"""Milvus client implementation."""

import asyncio
from typing import Any, cast

from typing_extensions import Unpack

from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionAddParams,
    BaseCollectionParams,
    BaseCollectionSearchParams,
)
from crewai.rag.core.exceptions import ClientMethodMismatchError
from crewai.rag.milvus.types import (
    AsyncEmbeddingFunction,
    EmbeddingFunction,
    MilvusCollectionCreateParams,
)
from crewai.rag.milvus.utils import (
    _ensure_collection,
    _ensure_list_embedding,
    _is_async_embedding_function,
    _is_sync_client,
    _prepare_documents_for_milvus,
    _prepare_search_params,
    _process_search_results,
    _validate_collection_schema,
    _validate_metric_type,
)
from crewai.rag.types import SearchResult


class MilvusClient(BaseClient):
    """Milvus implementation of the BaseClient protocol."""

    def __init__(
        self,
        client: Any,
        embedding_function: EmbeddingFunction | AsyncEmbeddingFunction,
        default_limit: int = 5,
        default_score_threshold: float = 0.6,
        default_batch_size: int = 100,
        dimension: int = 1536,
        metric_type: str = "COSINE",
        consistency_level: str | None = None,
    ) -> None:
        """Initialize MilvusClient with client and embedding function."""
        self.client = client
        self.embedding_function = embedding_function
        self.default_limit = default_limit
        self.default_score_threshold = default_score_threshold
        self.default_batch_size = default_batch_size
        self.dimension = dimension
        self.metric_type = _validate_metric_type(metric_type)
        self.consistency_level = consistency_level

    def create_collection(self, **kwargs: Unpack[MilvusCollectionCreateParams]) -> None:
        """Create a new collection in Milvus."""
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="create_collection",
                expected_client="MilvusClient",
                alt_method="acreate_collection",
                alt_client="MilvusClient",
            )

        collection_name = kwargs["collection_name"]
        if self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")

        dimension = kwargs.get("dimension", self.dimension)
        metric_type = _validate_metric_type(kwargs.get("metric_type", self.metric_type))
        consistency_level = kwargs.get("consistency_level", self.consistency_level)
        _ensure_collection(
            client=self.client,
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type,
            consistency_level=consistency_level,
        )

    async def acreate_collection(
        self, **kwargs: Unpack[MilvusCollectionCreateParams]
    ) -> None:
        """Create a new collection in Milvus asynchronously."""
        await asyncio.to_thread(self.create_collection, **kwargs)

    def get_or_create_collection(
        self, **kwargs: Unpack[MilvusCollectionCreateParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist."""
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="get_or_create_collection",
                expected_client="MilvusClient",
                alt_method="aget_or_create_collection",
                alt_client="MilvusClient",
            )

        collection_name = kwargs["collection_name"]
        dimension = kwargs.get("dimension", self.dimension)
        metric_type = _validate_metric_type(kwargs.get("metric_type", self.metric_type))
        consistency_level = kwargs.get("consistency_level", self.consistency_level)
        _ensure_collection(
            client=self.client,
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type,
            consistency_level=consistency_level,
        )
        return self.client.describe_collection(collection_name=collection_name)

    async def aget_or_create_collection(
        self, **kwargs: Unpack[MilvusCollectionCreateParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist asynchronously."""
        return await asyncio.to_thread(self.get_or_create_collection, **kwargs)

    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection."""
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="add_documents",
                expected_client="MilvusClient",
                alt_method="aadd_documents",
                alt_client="MilvusClient",
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]
        batch_size = kwargs.get("batch_size", self.default_batch_size)

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        _validate_collection_schema(
            client=self.client,
            collection_name=collection_name,
            dimension=self.dimension,
        )

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : min(i + batch_size, len(documents))]
            embeddings: list[list[float]] = []
            for doc in batch_docs:
                if _is_async_embedding_function(self.embedding_function):
                    raise TypeError(
                        "Async embedding function cannot be used with sync "
                        "add_documents. Use aadd_documents instead."
                    )
                sync_fn = cast(EmbeddingFunction, self.embedding_function)
                embeddings.append(_ensure_list_embedding(sync_fn(doc["content"])))

            data = _prepare_documents_for_milvus(
                documents=batch_docs,
                embeddings=embeddings,
            )
            self.client.upsert(collection_name=collection_name, data=data)

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection asynchronously."""
        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]
        batch_size = kwargs.get("batch_size", self.default_batch_size)

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        _validate_collection_schema(
            client=self.client,
            collection_name=collection_name,
            dimension=self.dimension,
        )

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : min(i + batch_size, len(documents))]
            embeddings: list[list[float]] = []
            for doc in batch_docs:
                if _is_async_embedding_function(self.embedding_function):
                    embeddings.append(
                        _ensure_list_embedding(
                            await self.embedding_function(doc["content"])
                        )
                    )
                else:
                    sync_fn = cast(EmbeddingFunction, self.embedding_function)
                    embeddings.append(_ensure_list_embedding(sync_fn(doc["content"])))

            data = _prepare_documents_for_milvus(
                documents=batch_docs,
                embeddings=embeddings,
            )
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=collection_name,
                data=data,
            )

    def search(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query."""
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="search",
                expected_client="MilvusClient",
                alt_method="asearch",
                alt_client="MilvusClient",
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", self.default_limit)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold", self.default_score_threshold)

        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        if _is_async_embedding_function(self.embedding_function):
            raise TypeError(
                "Async embedding function cannot be used with sync search. "
                "Use asearch instead."
            )
        sync_fn = cast(EmbeddingFunction, self.embedding_function)
        query_embedding = _ensure_list_embedding(sync_fn(query))

        search_kwargs = _prepare_search_params(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=limit,
            metric_type=self.metric_type,
            metadata_filter=metadata_filter,
        )
        response = self.client.search(**search_kwargs)
        return _process_search_results(
            response=response,
            metric_type=self.metric_type,
            score_threshold=score_threshold,
        )

    async def asearch(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously."""
        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", self.default_limit)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold", self.default_score_threshold)

        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        if _is_async_embedding_function(self.embedding_function):
            query_embedding = _ensure_list_embedding(
                await self.embedding_function(query)
            )
        else:
            sync_fn = cast(EmbeddingFunction, self.embedding_function)
            query_embedding = _ensure_list_embedding(sync_fn(query))

        search_kwargs = _prepare_search_params(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=limit,
            metric_type=self.metric_type,
            metadata_filter=metadata_filter,
        )
        response = await asyncio.to_thread(self.client.search, **dict(search_kwargs))
        return _process_search_results(
            response=response,
            metric_type=self.metric_type,
            score_threshold=score_threshold,
        )

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data."""
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="delete_collection",
                expected_client="MilvusClient",
                alt_method="adelete_collection",
                alt_client="MilvusClient",
            )

        collection_name = kwargs["collection_name"]

        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        self.client.drop_collection(collection_name=collection_name)

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data asynchronously."""
        await asyncio.to_thread(self.delete_collection, **kwargs)

    def reset(self) -> None:
        """Reset the vector database by deleting all collections and data."""
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="reset",
                expected_client="MilvusClient",
                alt_method="areset",
                alt_client="MilvusClient",
            )

        for collection_name in self.client.list_collections():
            self.client.drop_collection(collection_name=collection_name)

    async def areset(self) -> None:
        """Reset the vector database by deleting all collections asynchronously."""
        await asyncio.to_thread(self.reset)
