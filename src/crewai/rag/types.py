"""Type definitions for RAG (Retrieval-Augmented Generation) systems."""

from collections.abc import Callable, Mapping
from typing import TypeAlias, TypedDict, Any

from pydantic import BaseModel, Field
from typing_extensions import Required


class BaseRecord(TypedDict, total=False):
    """A typed dictionary representing a document record.

    This provides a simpler alternative to BaseDocument for cases where
    you want to pass document data as a dictionary.

    Attributes:
        doc_id: Optional unique identifier for the document. If not provided,
            a content-based ID will be generated using SHA256 hash.
        content: The text content of the document (required)
        metadata: Optional metadata associated with the document
    """

    doc_id: str
    content: Required[str]
    metadata: (
        Mapping[str, str | int | float | bool]
        | list[Mapping[str, str | int | float | bool]]
    )


# Vector types
DenseVector: TypeAlias = list[float]
IntVector: TypeAlias = list[int]


# TODO: move package specific types to their packages
# ChromaDB embedding types
# Excluding numpy array types for Pydantic compatibility
ChromaDbSingleEmbedding: TypeAlias = DenseVector | IntVector
ChromaDbEmbedding: TypeAlias = ChromaDbSingleEmbedding | list[DenseVector | IntVector]


# Qdrant embedding types
class QdrantSparseVector(BaseModel, extra="forbid"):
    """
    Sparse vector structure
    """

    indices: list[int] = Field(..., description="Indices must be unique")
    values: list[float] = Field(
        ..., description="Values and indices must be the same length"
    )


QdrantMultiVector: TypeAlias = list[DenseVector]
QdrantNamedVectors: TypeAlias = dict[
    str, DenseVector | QdrantSparseVector | QdrantMultiVector
]
QdrantEmbedding: TypeAlias = (
    DenseVector | QdrantMultiVector | QdrantNamedVectors | QdrantSparseVector
)

# Combined embedding types
Embedding: TypeAlias = ChromaDbEmbedding | QdrantEmbedding
EmbeddingFunction: TypeAlias = Callable[..., Any]
ChromaDbEmbeddingFunction: TypeAlias = Callable[
    [list[BaseRecord]], list[ChromaDbEmbedding]
]
QdrantEmbeddingFunction: TypeAlias = Callable[[list[BaseRecord]], list[QdrantEmbedding]]


class SearchResult(TypedDict):
    """Standard search result format for vector store queries.

    This provides a consistent interface for search results across different
    vector store implementations. Each implementation should convert their
    native result format to this standard format.

    Attributes:
        id: Unique identifier of the document
        content: The text content of the document
        metadata: Optional metadata associated with the document
        score: Similarity score (higher is better, typically between 0 and 1)
    """

    id: str
    content: str
    metadata: dict[str, Any]
    score: float
