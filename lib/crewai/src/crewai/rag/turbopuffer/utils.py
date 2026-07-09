"""Utility functions for turbopuffer operations."""

import asyncio
from typing import Any, TypeGuard
from uuid import uuid4

from turbopuffer import AsyncTurbopuffer, Turbopuffer

from crewai.rag.turbopuffer.constants import (
    CONTENT_KEY,
    NAMESPACE_PATTERN,
)
from crewai.rag.turbopuffer.types import (
    AsyncEmbeddingFunction,
    EmbeddingFunction,
    TurbopufferClientType,
)
from crewai.rag.types import BaseRecord, SearchResult


def _is_sync_client(client: TurbopufferClientType) -> TypeGuard[Turbopuffer]:
    """Type guard to check if the client is a synchronous Turbopuffer client.

    Args:
        client: The client to check.

    Returns:
        True if the client is a Turbopuffer (sync), False otherwise.
    """
    return isinstance(client, Turbopuffer)


def _is_async_client(client: TurbopufferClientType) -> TypeGuard[AsyncTurbopuffer]:
    """Type guard to check if the client is an asynchronous AsyncTurbopuffer client.

    Args:
        client: The client to check.

    Returns:
        True if the client is an AsyncTurbopuffer, False otherwise.
    """
    return isinstance(client, AsyncTurbopuffer)


def _is_async_embedding_function(
    func: EmbeddingFunction | AsyncEmbeddingFunction,
) -> TypeGuard[AsyncEmbeddingFunction]:
    """Type guard to check if the embedding function is async.

    Args:
        func: The embedding function to check.

    Returns:
        True if the function is async, False otherwise.
    """
    return asyncio.iscoroutinefunction(func)


def _validate_namespace_name(name: str) -> None:
    """Validate that a namespace name matches turbopuffer's constraints.

    Args:
        name: The namespace name to validate.

    Raises:
        ValueError: If the name doesn't match the pattern [A-Za-z0-9-_.]{1,128}.
    """
    if not NAMESPACE_PATTERN.match(name):
        raise ValueError(
            f"Invalid namespace name '{name}'. "
            "Must match pattern [A-Za-z0-9-_.] and be 1-128 characters."
        )


def _build_upsert_row(doc: BaseRecord, embedding: list[float]) -> dict[str, Any]:
    """Convert a BaseRecord and its embedding into a turbopuffer upsert row.

    Args:
        doc: Document dictionary containing content, metadata, and optional doc_id.
        embedding: The embedding vector for the document content.

    Returns:
        Dictionary ready to be passed to turbopuffer's write() as an upsert row.
    """
    doc_id = doc.get("doc_id", str(uuid4()))

    row: dict[str, Any] = {
        "id": doc_id,
        "vector": embedding,
        CONTENT_KEY: doc["content"],
    }

    metadata = doc.get("metadata", {})
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
    elif not isinstance(metadata, dict):
        metadata = dict(metadata) if metadata else {}

    row.update(metadata)

    return row


def _normalize_turbopuffer_score(dist: float) -> float:
    """Normalize a turbopuffer cosine distance to a [0, 1] similarity score.

    Args:
        dist: Raw cosine distance from turbopuffer (range [0, 2], lower is closer).

    Returns:
        Similarity score in [0, 1] where 1 is most similar.
    """
    # cosine_distance range is [0, 2]
    return max(0.0, min(1.0, 1.0 - (dist / 2.0)))


def _process_search_results(
    rows: list[Any],
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """Process turbopuffer query result rows into SearchResult format.

    Args:
        rows: Result rows from turbopuffer query.
        score_threshold: Optional minimum score to include in results.

    Returns:
        List of SearchResult dictionaries.
    """
    results: list[SearchResult] = []
    for row in rows:
        row_dict: dict[str, Any] = (
            row.to_dict() if hasattr(row, "to_dict") else dict(row)
        )

        dist = row_dict.get("$dist", 0.0)
        score = _normalize_turbopuffer_score(dist)

        if score_threshold is not None and score < score_threshold:
            continue

        doc_id = str(row_dict.get("id", ""))
        content = str(row_dict.get(CONTENT_KEY, ""))

        skip_keys = {"id", "vector", CONTENT_KEY, "$dist"}
        metadata: dict[str, Any] = {
            key: value for key, value in row_dict.items() if key not in skip_keys
        }

        result: SearchResult = {
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "score": score,
        }
        results.append(result)

    return results


def _build_metadata_filter(
    metadata_filter: dict[str, Any],
) -> tuple[str, str, Any] | tuple[str, list[tuple[str, str, Any]]]:
    """Convert a crewAI metadata filter dict to turbopuffer filter format.

    Args:
        metadata_filter: Dictionary of key-value pairs to filter on.

    Returns:
        Single condition tuple for one filter, or ("And", [...]) for multiple.
    """
    conditions = []
    for key, value in metadata_filter.items():
        conditions.append((key, "Eq", value))

    if len(conditions) == 1:
        return conditions[0]

    return ("And", conditions)
