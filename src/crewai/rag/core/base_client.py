"""Protocol for vector database client implementations."""
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable, TypedDict
from typing_extensions import Unpack, Required

from pydantic import BaseModel, Field

from crewai.rag.storage.document import VectorDocument
from crewai.rag.types import (
    EmbeddingFunction,
)


class VectorSearchResult(BaseModel):
    """Standardized vector search result.

    This class represents a single search result from a vector similarity search.
    It combines a VectorDocument with its similarity score.

    Attributes:
        document: The matched VectorDocument containing id, content, embedding, and metadata.
        score: Similarity score between the query and this document. Higher
            scores indicate greater similarity. The exact range depends on
            the distance metric used by the vector database.
    """

    document: VectorDocument = Field(description="The matched vector document")
    score: float = Field(description="Similarity score between the query and this document. Higher scores indicate greater similarity")


class CollectionParams(TypedDict, total=False):
    collection_name: Required[str]
    dimension: int


@runtime_checkable
class BaseClient(Protocol):
    """Protocol for vector store client implementations.

    This protocol defines the interface that all vector store client implementations
    must follow. It provides a consistent API for storing and retrieving
    documents with their vector embeddings across different vector database
    backends (e.g., Qdrant, ChromaDB, Weaviate). Implementing classes should
    handle connection management, data persistence, and vector similarity
    search operations specific to their backend.

    Notes:
      This protocol is planned to replace BaseRAGStorage
      (`src/crewai/rag/core/base_rag_storage.py`) in future versions.

    Attributes:
        embedding_function: Callable that takes a list of BaseDocument
            objects and returns a list of embedding vectors. Implementations
            should always provide a default embedding function.
    """

    embedding_function: EmbeddingFunction

    @abstractmethod
    def create_collection(
        self,
        **kwargs: Unpack[CollectionParams]
    ) -> None:
        """Create a new collection/index in the vector database.

        Keyword Args:
            collection_name: The name of the collection to create. Must be unique within
                the vector database instance.
            dimension: The dimensionality of the vectors to be stored. If not
                provided, the implementation may infer it from the embedding
                provider or raise an error.

        Raises:
            ValueError: If dimension is required but not provided, or if
                collection name already exists.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    async def acreate_collection(
        self,
        **kwargs: Unpack[CollectionParams]
    ) -> None:
        """Create a new collection/index in the vector database asynchronously.

        Keyword Args:
            collection_name: The name of the collection to create.
            dimension: The dimensionality of the vectors to be stored.

        Raises:
            ValueError: If dimension is required but not provided, or if
                collection name already exists.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    def get_or_create_collection(
        self,
        **kwargs: Unpack[CollectionParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist.

        This method provides a convenient way to ensure a collection exists
        without having to check for its existence first.

        Keyword Args:
            collection_name: The name of the collection to get or create.
            dimension: The dimensionality of vectors if creating a new collection.
                Ignored if collection already exists.

        Returns:
            A collection object whose type depends on the backend implementation.
            This could be a collection reference, ID, or client object.

        Raises:
            ValueError: If dimension is required for new collection but not provided.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    async def aget_or_create_collection(
        self,
        **kwargs: Unpack[CollectionParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist asynchronously.

        Keyword Args:
            collection_name: The name of the collection to get or create.
            dimension: The dimensionality of vectors if creating a new collection.

        Returns:
            A collection object whose type depends on the backend implementation.

        Raises:
            ValueError: If dimension is required for new collection but not provided.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    def add(
        self,
        collection_name: str,
        documents: list[VectorDocument],
    ) -> None:
        """Add documents with their embeddings to a collection.

        This method performs an upsert operation - if a document with the same ID
        already exists, it will be updated with the new embedding and metadata.

        Args:
            collection_name: The name of the collection to add documents to.
            documents: List of VectorDocument objects containing id, content,
                embedding, and metadata.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            TypeError: If documents are not VectorDocument instances.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>> from crewai.rag.storage.content import TextContent
            >>> store = QdrantVectorStore()
            >>> docs = [
            ...     VectorDocument(
            ...         doc_id="doc1",
            ...         content=TextContent(data="Hello world"),
            ...         embedding=[0.1, 0.2, ...],
            ...         metadata={"source": "file1"}
            ...     ),
            ...     VectorDocument(
            ...         doc_id="doc2",
            ...         content=TextContent(data="Python is great"),
            ...         embedding=[0.3, 0.4, ...],
            ...         metadata={"source": "file2"}
            ...     )
            ... ]
            >>> store.add(collection_name="my_docs", documents=docs)
        """
        ...

    @abstractmethod
    async def aadd(
        self,
        collection_name: str,
        documents: list[VectorDocument],
    ) -> None:
        """Add documents with their embeddings to a collection asynchronously.

        Args:
            collection_name: The name of the collection to add documents to.
            documents: List of VectorDocument objects containing id, content,
                embedding, and metadata.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            TypeError: If documents are not VectorDocument instances.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>> from crewai.rag.storage.document import VectorDocument
            >>> from crewai.rag.storage.content import TextContent
            >>>
            >>> async def add_documents():
            ...     store = QdrantVectorStore()
            ...     docs = [
            ...         VectorDocument(
            ...             doc_id="doc1",
            ...             content=TextContent(data="Hello world"),
            ...             embedding=[0.1, 0.2, ...],
            ...             metadata={"source": "file1"}
            ...         ),
            ...         VectorDocument(
            ...             doc_id="doc2",
            ...             content=TextContent(data="Python is great"),
            ...             embedding=[0.3, 0.4, ...],
            ...             metadata={"source": "file2"}
            ...         )
            ...     ]
            ...     await store.aadd(collection_name="my_docs", documents=docs)
            ...
            >>> asyncio.run(add_documents())
        """
        ...

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar documents using a query embedding.

        Performs a vector similarity search to find the most similar documents
        to the provided query embedding.

        Args:
            collection_name: The name of the collection to search in.
            query_embedding: The embedding vector to search for. Should have
                the same dimensionality as the vectors in the collection.
            limit: Maximum number of results to return. Defaults to 10.
            metadata_filter: Optional metadata filter to apply to the search. The exact
                format depends on the backend, but typically supports equality
                and range queries on metadata fields.
            score_threshold: Optional minimum similarity score threshold. Only
                results with scores >= this threshold will be returned. The
                score interpretation depends on the distance metric used.

        Returns:
            A list of VectorSearchResult objects ordered by similarity score in
            descending order. Each result contains doc_id, score, document text,
            and metadata.

        Raises:
            ValueError: If collection doesn't exist or query embedding has wrong dimensions.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>> store = QdrantVectorStore()
            >>> results = store.search(
            ...     collection_name="my_docs",
            ...     query_embedding=[0.1, 0.2, ...],
            ...     limit=5,
            ...     metadata_filter={"source": "file1"},
            ...     score_threshold=0.7
            ... )
            >>> for result in results:
            ...     print(f"{result.document.doc_id}: {result.score:.2f}")
        """
        ...

    @abstractmethod
    async def asearch(
        self,
        collection_name: str,
        query_embedding: list[float],
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar documents using a query embedding asynchronously.

        Args:
            collection_name: The name of the collection to search in.
            query_embedding: The embedding vector to search for.
            limit: Maximum number of results to return.
            metadata_filter: Optional metadata filter to apply to the search.
            score_threshold: Optional minimum similarity score threshold.

        Returns:
            A list of VectorSearchResult objects ordered by similarity score.

        Raises:
            ValueError: If collection doesn't exist or query embedding has wrong dimensions.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>>
            >>> async def search_documents():
            ...     store = QdrantVectorStore()
            ...     results = await store.asearch(
            ...         collection_name="my_docs",
            ...         query_embedding=[0.1, 0.2, ...],
            ...         limit=5,
            ...         metadata_filter={"source": "file1"},
            ...         score_threshold=0.7
            ...     )
            ...     for result in results:
            ...         print(f"{result.document.doc_id}: {result.score:.2f}")
            ...
            >>> asyncio.run(search_documents())
        """
        ...

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its data.

        This operation is irreversible and will permanently remove all documents,
        embeddings, and metadata associated with the collection.

        Args:
            collection_name: The name of the collection to delete.

        Raises:
            ValueError: If the collection doesn't exist.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>> store = QdrantVectorStore()
            >>> # Delete a specific collection
            >>> store.delete_collection("old_docs")
            >>> print("Collection 'old_docs' deleted successfully")
        """
        ...

    @abstractmethod
    async def adelete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its data asynchronously.

        Args:
            collection_name: The name of the collection to delete.

        Raises:
            ValueError: If the collection doesn't exist.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>>
            >>> async def delete_old_collection():
            ...     store = QdrantVectorStore()
            ...     await store.adelete_collection("old_docs")
            ...     print("Collection 'old_docs' deleted successfully")
            ...
            >>> asyncio.run(delete_old_collection())
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the vector database by deleting all collections and data.

        This method provides a way to completely clear the vector database,
        removing all collections and their contents. Use with caution as
        this operation is irreversible.

        Raises:
            ConnectionError: If unable to connect to the vector database backend.
            PermissionError: If the operation is not allowed by the backend.

        Example:
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>> store = QdrantVectorStore()
            >>> # WARNING: This will delete ALL data in the vector database
            >>> store.reset()
            >>> print("Vector database completely reset - all data deleted")
        """
        ...

    @abstractmethod
    async def areset(self) -> None:
        """Reset the vector database by deleting all collections and data asynchronously.

        Raises:
            ConnectionError: If unable to connect to the vector database backend.
            PermissionError: If the operation is not allowed by the backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>>
            >>> async def reset_store():
            ...     store = QdrantVectorStore()
            ...     # WARNING: This will delete ALL data in the vector store
            ...     await store.areset()
            ...     print("Vector store completely reset - all data deleted")
            ...
            >>> asyncio.run(reset_store())
        """
        ...

    @abstractmethod
    def get_embedding_dimension(self, sample_text: str = "test") -> int:
        """Get the dimension of embeddings used by this vector database client.

        This method helps determine the expected dimensionality of embeddings,
        either from the configured embedding provider or from existing collections.

        Args:
            sample_text: A sample text to use for generating a test embedding
                if needed. Defaults to "test".

        Returns:
            The dimension (number of components) of the embedding vectors.

        Raises:
            ValueError: If unable to determine embedding dimension (no embedding
                provider configured and no existing collections).
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>> store = QdrantVectorStore()
            >>> # Get embedding dimension using default sample text
            >>> dim = store.get_embedding_dimension()
            >>> print(f"Embedding dimension: {dim}")
            Embedding dimension: 768
            >>>
            >>> # Or with custom sample text
            >>> dim = store.get_embedding_dimension("Custom sample text")
            >>> print(f"Embedding dimension: {dim}")
            Embedding dimension: 768
        """
        ...

    @abstractmethod
    async def aget_embedding_dimension(self, sample_text: str = "test") -> int:
        """Get the dimension of embeddings used by this vector database client asynchronously.

        Args:
            sample_text: A sample text to use for generating a test embedding.

        Returns:
            The dimension (number of components) of the embedding vectors.

        Raises:
            ValueError: If unable to determine embedding dimension.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.storage.qdrant_vector_store import QdrantVectorStore
            >>>
            >>> async def check_dimensions():
            ...     store = QdrantVectorStore()
            ...     # Get embedding dimension using default sample text
            ...     dim = await store.aget_embedding_dimension()
            ...     print(f'Embedding dimension: {dim}')
            ...
            ...     # Or with custom sample text
            ...     dim = await store.aget_embedding_dimension("Custom sample text")
            ...     print(f"Embedding dimension: {dim}")
            ...
            >>> asyncio.run(check_dimensions())
            Embedding dimension: 768
            Embedding dimension: 768
        """
        ...
