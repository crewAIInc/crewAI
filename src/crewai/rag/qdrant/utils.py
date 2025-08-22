"""Utility functions for Qdrant operations."""

from typing import Any, TypeGuard
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, QdrantClient as SyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from crewai.rag.qdrant.types import (
    FilterCondition,
    PreparedSearchParams,
    QdrantClientType,
)
from crewai.rag.types import SearchResult, BaseRecord


def _is_sync_client(client: QdrantClientType) -> TypeGuard[SyncQdrantClient]:
    """Type guard to check if the client is a synchronous QdrantClient.

    Args:
        client: The client to check.

    Returns:
        True if the client is a QdrantClient, False otherwise.
    """
    return isinstance(client, SyncQdrantClient)


def _is_async_client(client: QdrantClientType) -> TypeGuard[AsyncQdrantClient]:
    """Type guard to check if the client is an asynchronous AsyncQdrantClient.

    Args:
        client: The client to check.

    Returns:
        True if the client is an AsyncQdrantClient, False otherwise.
    """
    return isinstance(client, AsyncQdrantClient)


def _prepare_search_params(
    collection_name: str,
    query_embedding: Any,
    limit: int,
    score_threshold: float | None,
    metadata_filter: dict[str, Any] | None,
) -> PreparedSearchParams:
    """Prepare search parameters for Qdrant query_points.

    Args:
        collection_name: Name of the collection to search.
        query_embedding: Embedding vector for the query.
        limit: Maximum number of results.
        score_threshold: Optional minimum similarity score.
        metadata_filter: Optional metadata filters.

    Returns:
        Dictionary of parameters for query_points method.
    """
    if not isinstance(query_embedding, list):
        query_embedding = query_embedding.tolist()

    search_kwargs: PreparedSearchParams = {
        "collection_name": collection_name,
        "query": query_embedding,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }

    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    if metadata_filter:
        filter_conditions: list[FilterCondition] = []
        for key, value in metadata_filter.items():
            filter_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

        search_kwargs["query_filter"] = Filter(must=filter_conditions)

    return search_kwargs


def _process_search_results(response: Any) -> list[SearchResult]:
    """Process Qdrant search response into SearchResult format.

    Args:
        response: Response from Qdrant query_points method.

    Returns:
        List of SearchResult dictionaries.
    """
    results: list[SearchResult] = []
    for point in response.points:
        result: SearchResult = {
            "id": str(point.id),
            "content": point.payload.get("content", ""),
            "metadata": {k: v for k, v in point.payload.items() if k != "content"},
            "score": point.score if point.score is not None else 0.0,
        }
        results.append(result)

    return results


def _create_point_from_document(doc: BaseRecord, embedding: Any) -> PointStruct:
    """Create a PointStruct from a document and its embedding.

    Args:
        doc: Document dictionary containing content, metadata, and optional doc_id.
        embedding: The embedding vector for the document content.

    Returns:
        PointStruct ready to be upserted to Qdrant.
    """
    doc_id = doc.get("doc_id", str(uuid4()))

    if not isinstance(embedding, list):
        embedding = embedding.tolist()

    metadata = doc.get("metadata", {})
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
    elif not isinstance(metadata, dict):
        metadata = dict(metadata) if metadata else {}

    return PointStruct(
        id=doc_id,
        vector=embedding,
        payload={"content": doc["content"], **metadata},
    )
