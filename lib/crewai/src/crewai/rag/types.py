"""Type definitions for RAG (Retrieval-Augmented Generation) systems."""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from typing_extensions import Required, TypedDict


class BaseRecord(TypedDict, total=False):
    """A typed dictionary representing a document record.

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


Embeddings: TypeAlias = list[list[float]]

EmbeddingFunction: TypeAlias = Callable[..., Any]


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
