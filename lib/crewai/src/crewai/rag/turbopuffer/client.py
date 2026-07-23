"""turbopuffer client implementation."""

from typing import Any, cast

from turbopuffer import omit
from typing_extensions import Unpack

from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionAddParams,
    BaseCollectionParams,
    BaseCollectionSearchParams,
)
from crewai.rag.core.exceptions import ClientMethodMismatchError
from crewai.rag.turbopuffer.constants import DistanceMetric
from crewai.rag.turbopuffer.types import (
    AsyncEmbeddingFunction,
    EmbeddingFunction,
    TurbopufferClientType,
)
from crewai.rag.turbopuffer.utils import (
    _build_metadata_filter,
    _build_upsert_row,
    _is_async_client,
    _is_async_embedding_function,
    _is_sync_client,
    _process_search_results,
    _validate_namespace_name,
)
from crewai.rag.types import SearchResult


class TurbopufferClient(BaseClient):
    """turbopuffer implementation of the BaseClient protocol.

    Provides vector database operations for turbopuffer, supporting both
    synchronous and asynchronous clients.

    Attributes:
        client: turbopuffer client instance (Turbopuffer or AsyncTurbopuffer).
        embedding_function: Function to generate embeddings for documents.
        default_limit: Default number of results to return in searches.
        default_score_threshold: Default minimum score for search results.
        distance_metric: Distance metric for similarity search.
    """

    def __init__(
        self,
        client: TurbopufferClientType,
        embedding_function: EmbeddingFunction | AsyncEmbeddingFunction,
        default_limit: int = 5,
        default_score_threshold: float = 0.6,
        default_batch_size: int = 100,
        distance_metric: DistanceMetric = "cosine_distance",
    ) -> None:
        """Initialize TurbopufferClient with client and embedding function.

        Args:
            client: Pre-configured turbopuffer client instance.
            embedding_function: Embedding function for text to vector conversion.
            default_limit: Default number of results to return in searches.
            default_score_threshold: Default minimum score for search results.
            default_batch_size: Default batch size for adding documents.
            distance_metric: Distance metric for similarity search.
        """
        self.client = client
        self.embedding_function = embedding_function
        self.default_limit = default_limit
        self.default_score_threshold = default_score_threshold
        self.default_batch_size = default_batch_size
        self.distance_metric = distance_metric

    def create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection (namespace) in turbopuffer.

        turbopuffer namespaces are created lazily on first write, so this
        method only validates the namespace name format.

        Keyword Args:
            collection_name: Name of the namespace to create.

        Raises:
            ClientMethodMismatchError: If called with an async client.
            ValueError: If namespace name is invalid.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="create_collection",
                expected_client="Turbopuffer",
                alt_method="acreate_collection",
                alt_client="AsyncTurbopuffer",
            )

        collection_name = kwargs["collection_name"]
        _validate_namespace_name(collection_name)

    async def acreate_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection (namespace) in turbopuffer asynchronously.

        Keyword Args:
            collection_name: Name of the namespace to create.

        Raises:
            ClientMethodMismatchError: If called with a sync client.
            ValueError: If namespace name is invalid.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="acreate_collection",
                expected_client="AsyncTurbopuffer",
                alt_method="create_collection",
                alt_client="Turbopuffer",
            )

        collection_name = kwargs["collection_name"]
        _validate_namespace_name(collection_name)

    def get_or_create_collection(
        self, **kwargs: Unpack[BaseCollectionParams]
    ) -> Any:
        """Get an existing namespace or prepare it for creation.

        turbopuffer namespaces are created lazily, so this validates
        the name and returns the namespace name.

        Keyword Args:
            collection_name: Name of the namespace.

        Returns:
            The namespace name string.

        Raises:
            ClientMethodMismatchError: If called with an async client.
            ValueError: If namespace name is invalid.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="get_or_create_collection",
                expected_client="Turbopuffer",
                alt_method="aget_or_create_collection",
                alt_client="AsyncTurbopuffer",
            )

        collection_name = kwargs["collection_name"]
        _validate_namespace_name(collection_name)
        return collection_name

    async def aget_or_create_collection(
        self, **kwargs: Unpack[BaseCollectionParams]
    ) -> Any:
        """Get an existing namespace or prepare it for creation asynchronously.

        Keyword Args:
            collection_name: Name of the namespace.

        Returns:
            The namespace name string.

        Raises:
            ClientMethodMismatchError: If called with a sync client.
            ValueError: If namespace name is invalid.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="aget_or_create_collection",
                expected_client="AsyncTurbopuffer",
                alt_method="get_or_create_collection",
                alt_client="Turbopuffer",
            )

        collection_name = kwargs["collection_name"]
        _validate_namespace_name(collection_name)
        return collection_name

    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a namespace.

        Keyword Args:
            collection_name: The namespace to add documents to.
            documents: List of BaseRecord dicts containing document data.
            batch_size: Optional batch size for processing documents.

        Raises:
            ClientMethodMismatchError: If called with an async client.
            ValueError: If documents list is empty.
            TypeError: If async embedding function used with sync method.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="add_documents",
                expected_client="Turbopuffer",
                alt_method="aadd_documents",
                alt_client="AsyncTurbopuffer",
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]
        batch_size = kwargs.get("batch_size", self.default_batch_size)

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if _is_async_embedding_function(self.embedding_function):
            raise TypeError(
                "Async embedding function cannot be used with sync add_documents. "
                "Use aadd_documents instead."
            )
        sync_fn = cast(EmbeddingFunction, self.embedding_function)

        ns = self.client.namespace(collection_name)

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : min(i + batch_size, len(documents))]
            upsert_rows = []
            for doc in batch_docs:
                embedding = sync_fn(doc["content"])
                row = _build_upsert_row(doc, embedding)
                upsert_rows.append(row)
            ns.write(
                upsert_rows=upsert_rows,
                distance_metric=self.distance_metric,
            )

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a namespace asynchronously.

        Keyword Args:
            collection_name: The namespace to add documents to.
            documents: List of BaseRecord dicts containing document data.
            batch_size: Optional batch size for processing documents.

        Raises:
            ClientMethodMismatchError: If called with a sync client.
            ValueError: If documents list is empty.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="aadd_documents",
                expected_client="AsyncTurbopuffer",
                alt_method="add_documents",
                alt_client="Turbopuffer",
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]
        batch_size = kwargs.get("batch_size", self.default_batch_size)

        if not documents:
            raise ValueError("Documents list cannot be empty")

        ns = self.client.namespace(collection_name)

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : min(i + batch_size, len(documents))]
            upsert_rows = []
            for doc in batch_docs:
                if _is_async_embedding_function(self.embedding_function):
                    async_fn = cast(AsyncEmbeddingFunction, self.embedding_function)
                    embedding = await async_fn(doc["content"])
                else:
                    sync_fn = cast(EmbeddingFunction, self.embedding_function)
                    embedding = sync_fn(doc["content"])
                row = _build_upsert_row(doc, embedding)
                upsert_rows.append(row)
            await ns.write(
                upsert_rows=upsert_rows,
                distance_metric=self.distance_metric,
            )

    def search(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query.

        Keyword Args:
            collection_name: Name of the namespace to search in.
            query: The text query to search for.
            limit: Maximum number of results to return.
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1).

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            ClientMethodMismatchError: If called with an async client.
            TypeError: If async embedding function used with sync method.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="search",
                expected_client="Turbopuffer",
                alt_method="asearch",
                alt_client="AsyncTurbopuffer",
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", self.default_limit)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold", self.default_score_threshold)

        if _is_async_embedding_function(self.embedding_function):
            raise TypeError(
                "Async embedding function cannot be used with sync search. "
                "Use asearch instead."
            )
        sync_fn = cast(EmbeddingFunction, self.embedding_function)
        query_embedding = sync_fn(query)

        ns = self.client.namespace(collection_name)

        query_kwargs: dict[str, Any] = {
            "rank_by": ("vector", "ANN", query_embedding),
            "top_k": limit,
            "exclude_attributes": ["vector"],
        }

        if metadata_filter:
            query_kwargs["filters"] = _build_metadata_filter(metadata_filter)
        else:
            query_kwargs["filters"] = omit

        result = ns.query(**query_kwargs)
        return _process_search_results(
            result.rows or [],
            score_threshold=score_threshold,
        )

    async def asearch(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously.

        Keyword Args:
            collection_name: Name of the namespace to search in.
            query: The text query to search for.
            limit: Maximum number of results to return.
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1).

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            ClientMethodMismatchError: If called with a sync client.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="asearch",
                expected_client="AsyncTurbopuffer",
                alt_method="search",
                alt_client="Turbopuffer",
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", self.default_limit)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold", self.default_score_threshold)

        if _is_async_embedding_function(self.embedding_function):
            async_fn = cast(AsyncEmbeddingFunction, self.embedding_function)
            query_embedding = await async_fn(query)
        else:
            sync_fn = cast(EmbeddingFunction, self.embedding_function)
            query_embedding = sync_fn(query)

        ns = self.client.namespace(collection_name)

        query_kwargs: dict[str, Any] = {
            "rank_by": ("vector", "ANN", query_embedding),
            "top_k": limit,
            "exclude_attributes": ["vector"],
        }

        if metadata_filter:
            query_kwargs["filters"] = _build_metadata_filter(metadata_filter)
        else:
            query_kwargs["filters"] = omit

        result = await ns.query(**query_kwargs)
        return _process_search_results(
            result.rows or [],
            score_threshold=score_threshold,
        )

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a namespace and all its data.

        Keyword Args:
            collection_name: Name of the namespace to delete.

        Raises:
            ClientMethodMismatchError: If called with an async client.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="delete_collection",
                expected_client="Turbopuffer",
                alt_method="adelete_collection",
                alt_client="AsyncTurbopuffer",
            )

        collection_name = kwargs["collection_name"]
        ns = self.client.namespace(collection_name)
        ns.delete_all()

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a namespace and all its data asynchronously.

        Keyword Args:
            collection_name: Name of the namespace to delete.

        Raises:
            ClientMethodMismatchError: If called with a sync client.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="adelete_collection",
                expected_client="AsyncTurbopuffer",
                alt_method="delete_collection",
                alt_client="Turbopuffer",
            )

        collection_name = kwargs["collection_name"]
        ns = self.client.namespace(collection_name)
        await ns.delete_all()

    def reset(self) -> None:
        """Reset by deleting all namespaces.

        Raises:
            ClientMethodMismatchError: If called with an async client.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="reset",
                expected_client="Turbopuffer",
                alt_method="areset",
                alt_client="AsyncTurbopuffer",
            )

        for ns in self.client.namespaces():
            self.client.namespace(ns.id).delete_all()

    async def areset(self) -> None:
        """Reset by deleting all namespaces asynchronously.

        Raises:
            ClientMethodMismatchError: If called with a sync client.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="areset",
                expected_client="AsyncTurbopuffer",
                alt_method="reset",
                alt_client="Turbopuffer",
            )

        async for ns in self.client.namespaces():
            await self.client.namespace(ns.id).delete_all()
