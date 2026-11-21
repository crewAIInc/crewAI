"""Utility functions for Qdrant operations."""

import asyncio
from collections.abc import Mapping
import json
import re
from typing import TypeGuard
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import (
    AsyncQdrantClient,
    QdrantClient as SyncQdrantClient,
)
from qdrant_client.models import (
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


# Characters Qdrant rejects in collection names (and that would otherwise become
# path separators in a local store), plus control characters.
_INVALID_COLLECTION_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_COLLECTION_NAME_BYTES = 255
# Payload key holding a document ID that had to be mapped to another point ID.
_DOC_ID_KEY = "crewai_doc_id"
# Namespace of the point IDs mapped from document IDs, apart from the default IDs.
_DOC_ID_NAMESPACE = uuid5(NAMESPACE_URL, "crewai:doc_id")


def _sanitize_collection_name(name: str) -> str:
    """Make a collection name valid for Qdrant.

    Replaces characters Qdrant rejects with ``_`` and limits the name to 255
    bytes. Names that are already valid are returned unchanged, so existing
    collections keep their names.

    Args:
        name: The requested collection name, e.g. one built from an agent role.

    Returns:
        A name accepted by both local and server Qdrant.
    """
    sanitized = _INVALID_COLLECTION_CHARS.sub("_", name)
    if sanitized in {".", ".."}:
        sanitized = sanitized.replace(".", "_")
    encoded = sanitized.encode("utf-8")
    if len(encoded) > _MAX_COLLECTION_NAME_BYTES:
        sanitized = encoded[:_MAX_COLLECTION_NAME_BYTES].decode("utf-8", "ignore")
    return sanitized


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
    metadata_filter: MetadataFilter | None,
) -> PreparedSearchParams:
    """Prepare search parameters for Qdrant query_points.

    Args:
        collection_name: Name of the collection to search.
        query_embedding: Embedding vector for the query.
        limit: Maximum number of results.
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


def _process_search_results(
    response: QueryResponse, score_threshold: float | None = None
) -> list[SearchResult]:
    """Process Qdrant search response into SearchResult format.

    The threshold is applied to the normalized score, the same 0-1 scale the
    results report and ChromaDB filters on. Qdrant's own ``score_threshold``
    compares the raw cosine similarity, so it is not sent with the query.

    Args:
        response: Response from Qdrant query_points method.
        score_threshold: Optional minimum normalized score (0-1) for results.

    Returns:
        List of SearchResult dictionaries.
    """
    results: list[SearchResult] = []
    for point in response.points:
        payload = dict(point.payload or {})
        doc_id = point.id
        stored = payload.get(_DOC_ID_KEY)
        if isinstance(stored, str) and stored != doc_id and _point_id(stored) == doc_id:
            doc_id = payload.pop(_DOC_ID_KEY)
        score = _normalize_qdrant_score(score=point.score)
        if score_threshold is not None and score < score_threshold:
            continue
        result: SearchResult = {
            "id": str(doc_id),
            "content": payload.get("content", ""),
            "metadata": {k: v for k, v in payload.items() if k != "content"},
            "score": score,
        }
        results.append(result)

    return results


def _default_doc_id(content: str, metadata: Mapping[str, object]) -> str:
    """Derive a stable point ID from a document's content and metadata.

    Saving the same document again then overwrites its point instead of adding a
    duplicate, matching how the ChromaDB client identifies documents.
    """
    key = content
    if metadata:
        key = f"{content}|{json.dumps(metadata, sort_keys=True, default=str)}"
    return str(uuid5(NAMESPACE_URL, key))


def _point_id(doc_id: str) -> str:
    """Return the point ID for a document ID.

    Qdrant accepts only integers and UUIDs as point IDs, so any other string is
    mapped to a UUID derived from it.
    """
    try:
        UUID(doc_id)
    except ValueError:
        return str(uuid5(_DOC_ID_NAMESPACE, doc_id))
    return doc_id


def _create_point_from_document(
    doc: BaseRecord, embedding: QueryEmbedding
) -> PointStruct:
    """Create a PointStruct from a document and its embedding.

    Args:
        doc: Document dictionary containing content, metadata, and optional doc_id.
            Without a doc_id, a deterministic ID is derived from content and metadata.
        embedding: The embedding vector for the document content.

    Returns:
        PointStruct ready to be upserted to Qdrant.
    """
    vector = _ensure_list_embedding(embedding)

    metadata = doc.get("metadata", {})
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
    elif not isinstance(metadata, dict):
        metadata = dict(metadata) if metadata else {}

    doc_id = doc.get("doc_id") or _default_doc_id(doc["content"], metadata)
    point_id = _point_id(doc_id) if isinstance(doc_id, str) else doc_id
    if point_id != doc_id:
        if _DOC_ID_KEY in metadata:
            raise ValueError(f"metadata key {_DOC_ID_KEY!r} is reserved for doc_id")
        metadata = {**metadata, _DOC_ID_KEY: doc_id}

    return PointStruct(
        id=point_id,
        vector=vector,
        payload={"content": doc["content"], **metadata},
    )
