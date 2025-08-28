"""Elasticsearch client implementation."""

from typing import Any, cast

from typing_extensions import Unpack

from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionParams,
    BaseCollectionAddParams,
    BaseCollectionSearchParams,
)
from crewai.rag.core.exceptions import ClientMethodMismatchError
from crewai.rag.elasticsearch.types import (
    AsyncEmbeddingFunction,
    EmbeddingFunction,
    ElasticsearchClientType,
    ElasticsearchCollectionCreateParams,
)
from crewai.rag.elasticsearch.utils import (
    _is_async_client,
    _is_async_embedding_function,
    _is_sync_client,
    _prepare_document_for_elasticsearch,
    _process_search_results,
    _build_vector_search_query,
    _get_index_mapping,
)
from crewai.rag.types import SearchResult


class ElasticsearchClient(BaseClient):
    """Elasticsearch implementation of the BaseClient protocol.

    Provides vector database operations for Elasticsearch, supporting both
    synchronous and asynchronous clients.

    Attributes:
        client: Elasticsearch client instance (Elasticsearch or AsyncElasticsearch).
        embedding_function: Function to generate embeddings for documents.
        vector_dimension: Dimension of the embedding vectors.
        similarity: Similarity function to use for vector search.
    """

    def __init__(
        self,
        client: ElasticsearchClientType,
        embedding_function: EmbeddingFunction | AsyncEmbeddingFunction,
        vector_dimension: int = 384,
        similarity: str = "cosine",
    ) -> None:
        """Initialize ElasticsearchClient with client and embedding function.

        Args:
            client: Pre-configured Elasticsearch client instance.
            embedding_function: Embedding function for text to vector conversion.
            vector_dimension: Dimension of the embedding vectors.
            similarity: Similarity function to use for vector search.
        """
        self.client = client
        self.embedding_function = embedding_function
        self.vector_dimension = vector_dimension
        self.similarity = similarity

    def create_collection(self, **kwargs: Unpack[ElasticsearchCollectionCreateParams]) -> None:
        """Create a new index in Elasticsearch.

        Keyword Args:
            collection_name: Name of the index to create. Must be unique.
            index_settings: Optional index settings.
            vector_dimension: Optional vector dimension override.
            similarity: Optional similarity function override.

        Raises:
            ValueError: If index with the same name already exists.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="create_collection",
                expected_client="Elasticsearch",
                alt_method="acreate_collection",
                alt_client="AsyncElasticsearch",
            )

        collection_name = kwargs["collection_name"]
        
        if self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' already exists")

        vector_dim = kwargs.get("vector_dimension", self.vector_dimension)
        similarity = kwargs.get("similarity", self.similarity)
        
        mapping = _get_index_mapping(vector_dim, similarity)
        
        index_settings = kwargs.get("index_settings", {})
        if index_settings:
            mapping["settings"] = index_settings

        self.client.indices.create(index=collection_name, body=mapping)

    async def acreate_collection(self, **kwargs: Unpack[ElasticsearchCollectionCreateParams]) -> None:
        """Create a new index in Elasticsearch asynchronously.

        Keyword Args:
            collection_name: Name of the index to create. Must be unique.
            index_settings: Optional index settings.
            vector_dimension: Optional vector dimension override.
            similarity: Optional similarity function override.

        Raises:
            ValueError: If index with the same name already exists.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="acreate_collection",
                expected_client="AsyncElasticsearch",
                alt_method="create_collection",
                alt_client="Elasticsearch",
            )

        collection_name = kwargs["collection_name"]
        
        if await self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' already exists")

        vector_dim = kwargs.get("vector_dimension", self.vector_dimension)
        similarity = kwargs.get("similarity", self.similarity)
        
        mapping = _get_index_mapping(vector_dim, similarity)
        
        index_settings = kwargs.get("index_settings", {})
        if index_settings:
            mapping["settings"] = index_settings

        await self.client.indices.create(index=collection_name, body=mapping)

    def get_or_create_collection(self, **kwargs: Unpack[ElasticsearchCollectionCreateParams]) -> Any:
        """Get an existing index or create it if it doesn't exist.

        Keyword Args:
            collection_name: Name of the index to get or create.
            index_settings: Optional index settings.
            vector_dimension: Optional vector dimension override.
            similarity: Optional similarity function override.

        Returns:
            Index info dict with name and other metadata.

        Raises:
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="get_or_create_collection",
                expected_client="Elasticsearch",
                alt_method="aget_or_create_collection",
                alt_client="AsyncElasticsearch",
            )

        collection_name = kwargs["collection_name"]

        if self.client.indices.exists(index=collection_name):
            return self.client.indices.get(index=collection_name)

        vector_dim = kwargs.get("vector_dimension", self.vector_dimension)
        similarity = kwargs.get("similarity", self.similarity)
        
        mapping = _get_index_mapping(vector_dim, similarity)
        
        index_settings = kwargs.get("index_settings", {})
        if index_settings:
            mapping["settings"] = index_settings

        self.client.indices.create(index=collection_name, body=mapping)
        return self.client.indices.get(index=collection_name)

    async def aget_or_create_collection(self, **kwargs: Unpack[ElasticsearchCollectionCreateParams]) -> Any:
        """Get an existing index or create it if it doesn't exist asynchronously.

        Keyword Args:
            collection_name: Name of the index to get or create.
            index_settings: Optional index settings.
            vector_dimension: Optional vector dimension override.
            similarity: Optional similarity function override.

        Returns:
            Index info dict with name and other metadata.

        Raises:
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="aget_or_create_collection",
                expected_client="AsyncElasticsearch",
                alt_method="get_or_create_collection",
                alt_client="Elasticsearch",
            )

        collection_name = kwargs["collection_name"]

        if await self.client.indices.exists(index=collection_name):
            return await self.client.indices.get(index=collection_name)

        vector_dim = kwargs.get("vector_dimension", self.vector_dimension)
        similarity = kwargs.get("similarity", self.similarity)
        
        mapping = _get_index_mapping(vector_dim, similarity)
        
        index_settings = kwargs.get("index_settings", {})
        if index_settings:
            mapping["settings"] = index_settings

        await self.client.indices.create(index=collection_name, body=mapping)
        return await self.client.indices.get(index=collection_name)

    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to an index.

        Keyword Args:
            collection_name: The name of the index to add documents to.
            documents: List of BaseRecord dicts containing document data.

        Raises:
            ValueError: If index doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="add_documents",
                expected_client="Elasticsearch",
                alt_method="aadd_documents",
                alt_client="AsyncElasticsearch",
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' does not exist")

        for doc in documents:
            if _is_async_embedding_function(self.embedding_function):
                raise TypeError(
                    "Async embedding function cannot be used with sync add_documents. "
                    "Use aadd_documents instead."
                )
            sync_fn = cast(EmbeddingFunction, self.embedding_function)
            embedding = sync_fn(doc["content"])
            prepared_doc = _prepare_document_for_elasticsearch(doc, embedding)
            
            self.client.index(
                index=collection_name,
                id=prepared_doc["id"],
                body=prepared_doc["body"]
            )

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to an index asynchronously.

        Keyword Args:
            collection_name: The name of the index to add documents to.
            documents: List of BaseRecord dicts containing document data.

        Raises:
            ValueError: If index doesn't exist or documents list is empty.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="aadd_documents",
                expected_client="AsyncElasticsearch",
                alt_method="add_documents",
                alt_client="Elasticsearch",
            )

        collection_name = kwargs["collection_name"]
        documents = kwargs["documents"]

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not await self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' does not exist")

        for doc in documents:
            if _is_async_embedding_function(self.embedding_function):
                async_fn = cast(AsyncEmbeddingFunction, self.embedding_function)
                embedding = await async_fn(doc["content"])
            else:
                sync_fn = cast(EmbeddingFunction, self.embedding_function)
                embedding = sync_fn(doc["content"])
            
            prepared_doc = _prepare_document_for_elasticsearch(doc, embedding)
            
            await self.client.index(
                index=collection_name,
                id=prepared_doc["id"],
                body=prepared_doc["body"]
            )

    def search(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query.

        Keyword Args:
            collection_name: Name of the index to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default: 10).
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1) for results.

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            ValueError: If index doesn't exist.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="search",
                expected_client="Elasticsearch",
                alt_method="asearch",
                alt_client="AsyncElasticsearch",
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", 10)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold")

        if not self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' does not exist")

        if _is_async_embedding_function(self.embedding_function):
            raise TypeError(
                "Async embedding function cannot be used with sync search. "
                "Use asearch instead."
            )
        sync_fn = cast(EmbeddingFunction, self.embedding_function)
        query_embedding = sync_fn(query)

        search_query = _build_vector_search_query(
            query_vector=query_embedding,
            limit=limit,
            metadata_filter=metadata_filter,
            score_threshold=score_threshold,
        )

        response = self.client.search(index=collection_name, body=search_query)
        return _process_search_results(response, score_threshold)

    async def asearch(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously.

        Keyword Args:
            collection_name: Name of the index to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default: 10).
            metadata_filter: Optional filter for metadata fields.
            score_threshold: Optional minimum similarity score (0-1) for results.

        Returns:
            List of SearchResult dicts containing id, content, metadata, and score.

        Raises:
            ValueError: If index doesn't exist.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="asearch",
                expected_client="AsyncElasticsearch",
                alt_method="search",
                alt_client="Elasticsearch",
            )

        collection_name = kwargs["collection_name"]
        query = kwargs["query"]
        limit = kwargs.get("limit", 10)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold")

        if not await self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' does not exist")

        if _is_async_embedding_function(self.embedding_function):
            async_fn = cast(AsyncEmbeddingFunction, self.embedding_function)
            query_embedding = await async_fn(query)
        else:
            sync_fn = cast(EmbeddingFunction, self.embedding_function)
            query_embedding = sync_fn(query)

        search_query = _build_vector_search_query(
            query_vector=query_embedding,
            limit=limit,
            metadata_filter=metadata_filter,
            score_threshold=score_threshold,
        )

        response = await self.client.search(index=collection_name, body=search_query)
        return _process_search_results(response, score_threshold)

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete an index and all its data.

        Keyword Args:
            collection_name: Name of the index to delete.

        Raises:
            ValueError: If index doesn't exist.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="delete_collection",
                expected_client="Elasticsearch",
                alt_method="adelete_collection",
                alt_client="AsyncElasticsearch",
            )

        collection_name = kwargs["collection_name"]

        if not self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' does not exist")

        self.client.indices.delete(index=collection_name)

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete an index and all its data asynchronously.

        Keyword Args:
            collection_name: Name of the index to delete.

        Raises:
            ValueError: If index doesn't exist.
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="adelete_collection",
                expected_client="AsyncElasticsearch",
                alt_method="delete_collection",
                alt_client="Elasticsearch",
            )

        collection_name = kwargs["collection_name"]

        if not await self.client.indices.exists(index=collection_name):
            raise ValueError(f"Index '{collection_name}' does not exist")

        await self.client.indices.delete(index=collection_name)

    def reset(self) -> None:
        """Reset the vector database by deleting all indices and data.

        Raises:
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_sync_client(self.client):
            raise ClientMethodMismatchError(
                method_name="reset",
                expected_client="Elasticsearch",
                alt_method="areset",
                alt_client="AsyncElasticsearch",
            )

        indices_response = self.client.indices.get(index="*")
        
        for index_name in indices_response.keys():
            if not index_name.startswith("."):
                self.client.indices.delete(index=index_name)

    async def areset(self) -> None:
        """Reset the vector database by deleting all indices and data asynchronously.

        Raises:
            ConnectionError: If unable to connect to Elasticsearch server.
        """
        if not _is_async_client(self.client):
            raise ClientMethodMismatchError(
                method_name="areset",
                expected_client="AsyncElasticsearch",
                alt_method="reset",
                alt_client="Elasticsearch",
            )

        indices_response = await self.client.indices.get(index="*")
        
        for index_name in indices_response.keys():
            if not index_name.startswith("."):
                await self.client.indices.delete(index=index_name)
