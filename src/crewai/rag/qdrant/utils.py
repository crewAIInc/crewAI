"""Utility functions for Qdrant operations."""

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from crewai.rag.qdrant.types import PreparedSearchParams
from crewai.rag.types import SearchResult


def prepare_search_params(
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
        filter_conditions = []
        for key, value in metadata_filter.items():
            filter_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

        search_kwargs["query_filter"] = Filter(must=filter_conditions)

    return search_kwargs


def process_search_results(response: Any) -> list[SearchResult]:
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
