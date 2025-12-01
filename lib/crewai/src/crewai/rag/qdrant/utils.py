"""Utility functions for Qdrant operations."""

import asyncio
from typing import TypeGuard
from uuid import uuid4

from qdrant_client import (
    AsyncQdrantClient,  # type: ignore[import-not-found]
    QdrantClient as SyncQdrantClient,  # type: ignore[import-not-found]
)
from qdrant_client.models import (  # type: ignore[import-not-found]
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    QueryResponse,
)

from crewai.rag.qdrant.constants import DEFAULT_VECTOR_PARAMS
from crewai.rag.qdrant.types import (
    AsyncEmbeddingFunction,
    CreateCollectionParams,
    EmbeddingFunction,
    FilterCondition,
    MetadataFilter,
    PreparedSearchParams,
    QdrantClientType,
    QdrantCollectionCreateParams,
    QueryEmbedding,
)
from crewai.rag.types import BaseRecord, SearchResult


def _ensure_list_embedding(embedding: QueryEmbedding) -> list[float]:
    """Convert embedding to list[float] format if needed.

    Args:
        embedding: Embedding vector as list or numpy array.

    Returns:
        Embedding as list[float].
    """
    if not isinstance(embedding, list):
        result = embedding.tolist()
        return result if isinstance(result, list) else [result]
    return embedding


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


def _get_collection_params(
    kwargs: QdrantCollectionCreateParams,
) -> CreateCollectionParams:
    """Extract collection creation parameters from kwargs."""
    params: CreateCollectionParams = {
        "collection_name": kwargs["collection_name"],
        "vectors_config": kwargs.get("vectors_config", DEFAULT_VECTOR_PARAMS),
    }

    if "sparse_vectors_config" in kwargs:
        params["sparse_vectors_config"] = kwargs["sparse_vectors_config"]
    if "shard_number" in kwargs:
        params["shard_number"] = kwargs["shard_number"]
    if "sharding_method" in kwargs:
        params["sharding_method"] = kwargs["sharding_method"]
    if "replication_factor" in kwargs:
        params["replication_factor"] = kwargs["replication_factor"]
    if "write_consistency_factor" in kwargs:
        params["write_consistency_factor"] = kwargs["write_consistency_factor"]
    if "on_disk_payload" in kwargs:
        params["on_disk_payload"] = kwargs["on_disk_payload"]
    if "hnsw_config" in kwargs:
        params["hnsw_config"] = kwargs["hnsw_config"]
    if "optimizers_config" in kwargs:
        params["optimizers_config"] = kwargs["optimizers_config"]
    if "wal_config" in kwargs:
        params["wal_config"] = kwargs["wal_config"]
    if "quantization_config" in kwargs:
        params["quantization_config"] = kwargs["quantization_config"]
    if "init_from" in kwargs:
        params["init_from"] = kwargs["init_from"]
    if "timeout" in kwargs:
        params["timeout"] = kwargs["timeout"]

    return params


def _prepare_search_params(
    collection_name: str,
    query_embedding: QueryEmbedding,
    limit: int,
    score_threshold: float | None,
    metadata_filter: MetadataFilter | None,
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
    query_vector = _ensure_list_embedding(query_embedding)

    search_kwargs: PreparedSearchParams = {
        "collection_name": collection_name,
        "query": query_vector,
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


def _normalize_qdrant_score(score: float) -> float:
    """Normalize Qdrant cosine similarity score to [0, 1] range.

    Converts from Qdrant's [-1, 1] cosine similarity range to [0, 1] range for standardization across clients.

    Args:
        score: Raw cosine similarity score from Qdrant [-1, 1].

    Returns:
        Normalized score in [0, 1] range where 1 is most similar.
    """
    normalized = (score + 1.0) / 2.0
    return max(0.0, min(1.0, normalized))


def _process_search_results(response: QueryResponse) -> list[SearchResult]:
    """Process Qdrant search response into SearchResult format.

    Args:
        response: Response from Qdrant query_points method.

    Returns:
        List of SearchResult dictionaries.
    """
    results: list[SearchResult] = []
    for point in response.points:
        payload = point.payload or {}
        score = _normalize_qdrant_score(score=point.score)
        result: SearchResult = {
            "id": str(point.id),
            "content": payload.get("content", ""),
            "metadata": {k: v for k, v in payload.items() if k != "content"},
            "score": score,
        }
        results.append(result)

    return results


def _create_point_from_document(
    doc: BaseRecord, embedding: QueryEmbedding
) -> PointStruct:
    """Create a PointStruct from a document and its embedding.

    Args:
        doc: Document dictionary containing content, metadata, and optional doc_id.
        embedding: The embedding vector for the document content.

    Returns:
        PointStruct ready to be upserted to Qdrant.
    """
    doc_id = doc.get("doc_id", str(uuid4()))
    vector = _ensure_list_embedding(embedding)

    metadata = doc.get("metadata", {})
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
    elif not isinstance(metadata, dict):
        metadata = dict(metadata) if metadata else {}

    return PointStruct(
        id=doc_id,
        vector=vector,
        payload={"content": doc["content"], **metadata},
    )
