"""Utility functions for ChromaDB client implementation."""

import hashlib
from collections.abc import Mapping
from typing import Literal, TypeGuard, cast

from chromadb.api import AsyncClientAPI, ClientAPI
from chromadb.api.types import (
    Include,
    IncludeEnum,
    QueryResult,
)
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.models.Collection import Collection
from crewai.rag.chromadb.constants import (
    DEFAULT_COLLECTION,
    INVALID_CHARS_PATTERN,
    IPV4_PATTERN,
    MAX_COLLECTION_LENGTH,
    MIN_COLLECTION_LENGTH,
)
from crewai.rag.chromadb.types import (
    ChromaDBClientType,
    ChromaDBCollectionSearchParams,
    ExtractedSearchParams,
    PreparedDocuments,
)
from crewai.rag.types import BaseRecord, SearchResult


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
) -> ExtractedSearchParams:
    """Extract search parameters from kwargs.

    Args:
        kwargs: Keyword arguments containing search parameters.

    Returns:
        ExtractedSearchParams with all extracted parameters.
    """
    return ExtractedSearchParams(
        collection_name=kwargs["collection_name"],
        query=kwargs["query"],
        limit=kwargs.get("limit", 10),
        metadata_filter=kwargs.get("metadata_filter"),
        score_threshold=kwargs.get("score_threshold"),
        where=kwargs.get("where"),
        where_document=kwargs.get("where_document"),
        include=kwargs.get(
            "include",
            [IncludeEnum.metadatas, IncludeEnum.documents, IncludeEnum.distances],
        ),
    )


def _convert_distance_to_score(
    distance: float,
    distance_metric: Literal["l2", "cosine", "ip"],
) -> float:
    """Convert ChromaDB distance to similarity score.

    Notes:
        Assuming all embedding are unit-normalized for now, including custom embeddings.

    Args:
        distance: The distance value from ChromaDB.
        distance_metric: The distance metric used ("l2", "cosine", or "ip").

    Returns:
        Similarity score in range [0, 1] where 1 is most similar.
    """
    if distance_metric == "cosine":
        score = 1.0 - 0.5 * distance
        return max(0.0, min(1.0, score))
    raise ValueError(f"Unsupported distance metric: {distance_metric}")


def _convert_chromadb_results_to_search_results(
    results: QueryResult,
    include: Include,
    distance_metric: Literal["l2", "cosine", "ip"],
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """Convert ChromaDB query results to SearchResult format.

    Args:
        results: ChromaDB query results.
        include: List of fields that were included in the query.
        distance_metric: The distance metric used by the collection.
        score_threshold: Optional minimum similarity score (0-1) for results.

    Returns:
        List of SearchResult dicts containing id, content, metadata, and score.
    """
    search_results: list[SearchResult] = []

    include_strings = [item.value for item in include]

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
        if not distances or i >= len(distances):
            continue

        distance = distances[i]
        score = _convert_distance_to_score(
            distance=distance, distance_metric=distance_metric
        )

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


def _process_query_results(
    collection: Collection | AsyncCollection,
    results: QueryResult,
    params: ExtractedSearchParams,
) -> list[SearchResult]:
    """Process ChromaDB query results and convert to SearchResult format.

    Args:
        collection: The ChromaDB collection (sync or async) that was queried.
        results: Raw query results from ChromaDB.
        params: The search parameters used for the query.

    Returns:
        List of SearchResult dicts containing id, content, metadata, and score.
    """

    distance_metric = cast(
        Literal["l2", "cosine", "ip"],
        collection.metadata.get("hnsw:space", "l2") if collection.metadata else "l2",
    )

    return _convert_chromadb_results_to_search_results(
        results=results,
        include=params.include,
        distance_metric=distance_metric,
        score_threshold=params.score_threshold,
    )


def _is_ipv4_pattern(name: str) -> bool:
    """Check if a string matches an IPv4 address pattern.

    Args:
        name: The string to check

    Returns:
        True if the string matches an IPv4 pattern, False otherwise
    """
    return bool(IPV4_PATTERN.match(name))


def _sanitize_collection_name(
    name: str | None, max_collection_length: int = MAX_COLLECTION_LENGTH
) -> str:
    """Sanitize a collection name to meet ChromaDB requirements.

    Requirements:
    1. 3-63 characters long
    2. Starts and ends with alphanumeric character
    3. Contains only alphanumeric characters, underscores, or hyphens
    4. No consecutive periods
    5. Not a valid IPv4 address

    Args:
        name: The original collection name to sanitize
        max_collection_length: Maximum allowed length for the collection name

    Returns:
        A sanitized collection name that meets ChromaDB requirements
    """
    if not name:
        return DEFAULT_COLLECTION

    if _is_ipv4_pattern(name):
        name = f"ip_{name}"

    sanitized = INVALID_CHARS_PATTERN.sub("_", name)

    if not sanitized[0].isalnum():
        sanitized = "a" + sanitized

    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + "z"

    if len(sanitized) < MIN_COLLECTION_LENGTH:
        sanitized = sanitized + "x" * (MIN_COLLECTION_LENGTH - len(sanitized))
    if len(sanitized) > max_collection_length:
        sanitized = sanitized[:max_collection_length]
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + "z"

    return sanitized
