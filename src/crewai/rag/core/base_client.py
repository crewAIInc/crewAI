"""Protocol for vector database client implementations."""

from abc import abstractmethod
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing_extensions import Required, TypedDict, Unpack

from crewai.rag.types import (
    BaseRecord,
    EmbeddingFunction,
    SearchResult,
)


class BaseCollectionParams(TypedDict):
    """Base parameters for collection operations.

    Attributes:
        collection_name: The name of the collection/index to operate on.
    """

    collection_name: Required[
        Annotated[
            str,
            "Name of the collection/index. Implementations may have specific constraints (e.g., character limits, allowed characters, case sensitivity).",
        ]
    ]


class BaseCollectionAddParams(BaseCollectionParams, total=False):
    """Parameters for adding documents to a collection.

    Extends BaseCollectionParams with document-specific fields.

    Attributes:
        collection_name: The name of the collection to add documents to.
        documents: List of BaseRecord dictionaries containing document data.
        batch_size: Optional batch size for processing documents to avoid token limits.
    """

    documents: Required[list[BaseRecord]]
    batch_size: int


class BaseCollectionSearchParams(BaseCollectionParams, total=False):
    """Parameters for searching within a collection.

    Extends BaseCollectionParams with search-specific optional fields.
    All fields except collection_name and query are optional.

    Attributes:
        query: The text query to search for (required).
        limit: Maximum number of results to return.
        metadata_filter: Filter results by metadata fields.
        score_threshold: Minimum similarity score for results (0-1).
    """

    query: Required[str]
    limit: int
    metadata_filter: dict[str, Any] | None
    score_threshold: float


@runtime_checkable
class BaseClient(Protocol):
    """Protocol for vector store client implementations.

    This protocol defines the interface that all vector store client implementations
    must follow. It provides a consistent API for storing and retrieving
    documents with their vector embeddings across different vector database
    backends (e.g., Qdrant, ChromaDB, Weaviate). Implementing classes should
    handle connection management, data persistence, and vector similarity
    search operations specific to their backend.

    Implementation Guidelines:
        Implementations should accept BaseClientParams in their constructor to allow
        passing pre-configured client instances:

        class MyVectorClient:
            def __init__(self, client: Any | None = None, **kwargs):
                if client:
                    self.client = client
                else:
                    self.client = self._create_default_client(**kwargs)

    Notes:
      This protocol replaces the former BaseRAGStorage abstraction,
      providing a cleaner interface for vector store operations.

    Attributes:
        embedding_function: Callable that takes a list of text strings
            and returns a list of embedding vectors. Implementations
            should always provide a default embedding function.
        client: The underlying vector database client instance. This could be
            passed via BaseClientParams during initialization or created internally.
    """

    client: Any
    embedding_function: EmbeddingFunction

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for BaseClient Protocol.

        This allows the Protocol to be used in Pydantic models without
        requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()

    @abstractmethod
    def create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection/index in the vector database.

        Keyword Args:
            collection_name: The name of the collection to create. Must be unique within
                the vector database instance.

        Raises:
            ValueError: If collection name already exists.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    async def acreate_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection/index in the vector database asynchronously.

        Keyword Args:
            collection_name: The name of the collection to create. Must be unique within
                the vector database instance.

        Raises:
            ValueError: If collection name already exists.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    def get_or_create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> Any:
        """Get an existing collection or create it if it doesn't exist.

        This method provides a convenient way to ensure a collection exists
        without having to check for its existence first.

        Keyword Args:
            collection_name: The name of the collection to get or create.

        Returns:
            A collection object whose type depends on the backend implementation.
            This could be a collection reference, ID, or client object.

        Raises:
            ValueError: If unable to create the collection.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    async def aget_or_create_collection(
        self, **kwargs: Unpack[BaseCollectionParams]
    ) -> Any:
        """Get an existing collection or create it if it doesn't exist asynchronously.

        Keyword Args:
            collection_name: The name of the collection to get or create.

        Returns:
            A collection object whose type depends on the backend implementation.

        Raises:
            ValueError: If unable to create the collection.
            ConnectionError: If unable to connect to the vector database backend.
        """
        ...

    @abstractmethod
    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection.

        This method performs an upsert operation - if a document with the same ID
        already exists, it will be updated with the new content and metadata.

        Implementations should handle embedding generation internally based on
        the configured embedding function.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing:
                - content: The text content (required)
                - doc_id: Optional unique identifier (auto-generated from content hash if missing)
                - metadata: Optional metadata dictionary
                Embeddings will be generated automatically.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            TypeError: If documents are not BaseRecord dict instances.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>> from crewai.rag.types import BaseRecord
            >>> client = ChromaDBClient()
            >>>
            >>> records: list[BaseRecord] = [
            ...     {
            ...         "content": "Machine learning basics",
            ...         "metadata": {"source": "file3", "topic": "ML"}
            ...     },
            ...     {
            ...         "doc_id": "custom_id",
            ...         "content": "Deep learning fundamentals",
            ...         "metadata": {"source": "file4", "topic": "DL"}
            ...     }
            ... ]
            >>> client.add_documents(collection_name="my_docs", documents=records)
            >>>
            >>> records_with_id: list[BaseRecord] = [
            ...     {
            ...         "doc_id": "nlp_001",
            ...         "content": "Advanced NLP techniques",
            ...         "metadata": {"source": "file5", "topic": "NLP"}
            ...     }
            ... ]
            >>> client.add_documents(collection_name="my_docs", documents=records_with_id)
        """
        ...

    @abstractmethod
    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection asynchronously.

        Implementations should handle embedding generation internally based on
        the configured embedding function.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing:
                - content: The text content (required)
                - doc_id: Optional unique identifier (auto-generated from content hash if missing)
                - metadata: Optional metadata dictionary
                Embeddings will be generated automatically.

        Raises:
            ValueError: If collection doesn't exist or documents list is empty.
            TypeError: If documents are not BaseRecord dict instances.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>> from crewai.rag.types import BaseRecord
            >>>
            >>> async def add_documents():
            ...     client = ChromaDBClient()
            ...
            ...     records: list[BaseRecord] = [
            ...         {
            ...             "doc_id": "doc2",
            ...             "content": "Async operations in Python",
            ...             "metadata": {"source": "file2", "topic": "async"}
            ...         }
            ...     ]
            ...     await client.aadd_documents(collection_name="my_docs", documents=records)
            ...
            >>> asyncio.run(add_documents())
        """
        ...

    @abstractmethod
    def search(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query.

        Performs a vector similarity search to find the most similar documents
        to the provided query.

        Keyword Args:
            collection_name: The name of the collection to search in.
            query: The text query to search for. The implementation handles
                embedding generation internally.
            limit: Maximum number of results to return. Defaults to 10.
            metadata_filter: Optional metadata filter to apply to the search. The exact
                format depends on the backend, but typically supports equality
                and range queries on metadata fields.
            score_threshold: Optional minimum similarity score threshold. Only
                results with scores >= this threshold will be returned. The
                score interpretation depends on the distance metric used.

        Returns:
            A list of SearchResult dictionaries ordered by similarity score in
            descending order. Each result contains:
                - id: Document ID
                - content: Document text content
                - metadata: Document metadata
                - score: Similarity score (0-1, higher is better)

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>> client = ChromaDBClient()
            >>>
            >>> results = client.search(
            ...     collection_name="my_docs",
            ...     query="What is machine learning?",
            ...     limit=5,
            ...     metadata_filter={"source": "file1"},
            ...     score_threshold=0.7
            ... )
            >>> for result in results:
            ...     print(f"{result['id']}: {result['score']:.2f}")
        """
        ...

    @abstractmethod
    async def asearch(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously.

        Keyword Args:
            collection_name: The name of the collection to search in.
            query: The text query to search for. The implementation handles
                embedding generation internally.
            limit: Maximum number of results to return. Defaults to 10.
            metadata_filter: Optional metadata filter to apply to the search.
            score_threshold: Optional minimum similarity score threshold.

        Returns:
            A list of SearchResult dictionaries ordered by similarity score.

        Raises:
            ValueError: If collection doesn't exist.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>>
            >>> async def search_documents():
            ...     client = ChromaDBClient()
            ...     results = await client.asearch(
            ...         collection_name="my_docs",
            ...         query="Python programming best practices",
            ...         limit=5,
            ...         metadata_filter={"source": "file1"},
            ...         score_threshold=0.7
            ...     )
            ...     for result in results:
            ...         print(f"{result['id']}: {result['score']:.2f}")
            ...
            >>> asyncio.run(search_documents())
        """
        ...

    @abstractmethod
    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data.

        This operation is irreversible and will permanently remove all documents,
        embeddings, and metadata associated with the collection.

        Keyword Args:
            collection_name: The name of the collection to delete.

        Raises:
            ValueError: If the collection doesn't exist.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>> client = ChromaDBClient()
            >>> client.delete_collection(collection_name="old_docs")
            >>> print("Collection 'old_docs' deleted successfully")
        """
        ...

    @abstractmethod
    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data asynchronously.

        Keyword Args:
            collection_name: The name of the collection to delete.

        Raises:
            ValueError: If the collection doesn't exist.
            ConnectionError: If unable to connect to the vector database backend.

        Example:
            >>> import asyncio
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>>
            >>> async def delete_old_collection():
            ...     client = ChromaDBClient()
            ...     await client.adelete_collection(collection_name="old_docs")
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
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>> client = ChromaDBClient()
            >>> client.reset()
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
            >>> from crewai.rag.chromadb.client import ChromaDBClient
            >>>
            >>> async def reset_database():
            ...     client = ChromaDBClient()
            ...     await client.areset()
            ...     print("Vector database completely reset - all data deleted")
            ...
            >>> asyncio.run(reset_database())
        """
        ...
