"""ChromaDB client implementation."""

import hashlib
from collections.abc import Mapping
from typing import Any, NamedTuple, TypeGuard

from chromadb.api import AsyncClientAPI, ClientAPI
from chromadb.api.configuration import CollectionConfigurationInterface
from chromadb.api.types import (
    CollectionMetadata,
    DataLoader,
    Embeddable,
    EmbeddingFunction as ChromaEmbeddingFunction,
    Include,
    IncludeEnum,
    Loadable,
    QueryResult,
    Where,
    WhereDocument,
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


class PreparedDocuments(NamedTuple):
    """Prepared documents ready for ChromaDB insertion.

    Attributes:
        ids: List of document IDs
        texts: List of document texts
        metadatas: List of document metadata mappings
    """

    ids: list[str]
    texts: list[str]
    metadatas: list[Mapping[str, str | int | float | bool]]


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


class ChromaDBCollectionSearchParams(BaseCollectionSearchParams, total=False):
    """Parameters for searching a ChromaDB collection.

    This class extends BaseCollectionSearchParams to include ChromaDB-specific
    search parameters like where clauses and include options.
    """

    where: Where
    where_document: WhereDocument
    include: Include


def _prepare_documents_for_chromadb(
    documents: list[BaseRecord],
) -> PreparedDocuments:
    """Prepare documents for ChromaDB by extracting IDs, texts, and metadata.

    Args:
        documents: List of BaseRecord documents to prepare.

    Returns:
        PreparedDocuments with ids, texts, and metadatas ready for ChromaDB.
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
                metadatas.append(metadata[0] if metadata else {})
            else:
                metadatas.append(metadata)
        else:
            metadatas.append({})

    return PreparedDocuments(ids, texts, metadatas)


def _extract_search_params(
    kwargs: ChromaDBCollectionSearchParams,
) -> tuple[
    str,
    str,
    int,
    dict[str, Any] | None,
    float | None,
    Where | None,
    WhereDocument | None,
    Include,
]:
    """Extract search parameters from kwargs.

    Args:
        kwargs: Keyword arguments containing search parameters.

    Returns:
        Tuple of (collection_name, query, limit, metadata_filter, score_threshold, where, where_document, include).
    """
    collection_name = kwargs["collection_name"]
    query = kwargs["query"]
    limit = kwargs.get("limit", 10)
    metadata_filter = kwargs.get("metadata_filter")
    score_threshold = kwargs.get("score_threshold")
    where = kwargs.get("where")
    where_document = kwargs.get("where_document")
    include = kwargs.get(
        "include", [IncludeEnum.metadatas, IncludeEnum.documents, IncludeEnum.distances]
    )

    return (
        collection_name,
        query,
        limit,
        metadata_filter,
        score_threshold,
        where,
        where_document,
        include,
    )


def _convert_chromadb_results_to_search_results(
    results: QueryResult,
    include: Include,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """Convert ChromaDB query results to SearchResult format.

    Args:
        results: ChromaDB query results.
        include: List of fields that were included in the query.
        score_threshold: Optional minimum similarity score (0-1) for results.

    Returns:
        List of SearchResult dicts containing id, content, metadata, and score.
    """
    search_results: list[SearchResult] = []

    include_strings = [
        item.value if isinstance(item, IncludeEnum) else item for item in include
    ]

    ids = results["ids"][0] if results.get("ids") else []

    documents_list = results.get("documents")
    documents = (
        documents_list[0] if documents_list and "documents" in include_strings else []
    )

    metadatas_list = results.get("metadatas")
    metadatas = (
        metadatas_list[0] if metadatas_list and "metadatas" in include_strings else []
    )

    distances_list = results.get("distances")
    distances = (
        distances_list[0] if distances_list and "distances" in include_strings else []
    )

    for i, doc_id in enumerate(ids):
        score = 1.0 - (distances[i] / 2.0) if distances and i < len(distances) else 0.0
        if score_threshold and score < score_threshold:
            continue

        result: SearchResult = {
            "id": doc_id,
            "content": documents[i] if documents and i < len(documents) else "",
            "metadata": dict(metadatas[i]) if metadatas and i < len(metadatas) else {},
            "score": score,
        }
        search_results.append(result)

    return search_results


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

        prepared = _prepare_documents_for_chromadb(documents)
        collection.add(
            ids=prepared.ids,
            documents=prepared.texts,
            metadatas=prepared.metadatas,
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
        prepared = _prepare_documents_for_chromadb(documents)
        await collection.add(
            ids=prepared.ids,
            documents=prepared.texts,
            metadatas=prepared.metadatas,
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

        (
            collection_name,
            query,
            limit,
            metadata_filter,
            score_threshold,
            where,
            where_document,
            include,
        ) = _extract_search_params(kwargs)

        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        if where is None and metadata_filter:
            where = metadata_filter

        results: QueryResult = collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,
            where_document=where_document,
            include=include,
        )

        return _convert_chromadb_results_to_search_results(
            results, include, score_threshold
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

        (
            collection_name,
            query,
            limit,
            metadata_filter,
            score_threshold,
            where,
            where_document,
            include,
        ) = _extract_search_params(kwargs)

        collection = await self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        if where is None and metadata_filter:
            where = metadata_filter

        results: QueryResult = await collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,
            where_document=where_document,
            include=include,
        )

        return _convert_chromadb_results_to_search_results(
            results, include, score_threshold
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
        self.client.delete_collection(name=collection_name)

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
        await self.client.delete_collection(name=collection_name)

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
