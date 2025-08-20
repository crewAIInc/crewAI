"""Type definitions specific to ChromaDB implementation."""

from collections.abc import Mapping
from typing import Any, NamedTuple

from chromadb.api import ClientAPI, AsyncClientAPI
from chromadb.api.types import Include, Where, WhereDocument

ChromaDBClientType = ClientAPI | AsyncClientAPI


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


class ExtractedSearchParams(NamedTuple):
    """Extracted search parameters for ChromaDB queries.

    Attributes:
        collection_name: Name of the collection to search
        query: Search query text
        limit: Maximum number of results
        metadata_filter: Optional metadata filter
        score_threshold: Optional minimum similarity score
        where: Optional ChromaDB where clause
        where_document: Optional ChromaDB document filter
        include: Fields to include in results
    """

    collection_name: str
    query: str
    limit: int
    metadata_filter: dict[str, Any] | None
    score_threshold: float | None
    where: Where | None
    where_document: WhereDocument | None
    include: Include
