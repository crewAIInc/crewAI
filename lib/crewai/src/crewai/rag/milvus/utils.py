"""Utility functions for Milvus operations."""

from __future__ import annotations

import asyncio
from functools import cache
import hashlib
import importlib.metadata
import json
import math
import re
from typing import Any, TypeGuard, cast

from pymilvus import (  # type: ignore[import-untyped]
    DataType,
    MilvusClient as SyncMilvusClient,
)

from crewai.rag.milvus.constants import (
    DEFAULT_CONTENT_MAX_LENGTH,
    DEFAULT_ID_MAX_LENGTH,
    VALID_METRIC_TYPES,
)
from crewai.rag.milvus.types import (
    AsyncEmbeddingFunction,
    EmbeddingFunction,
    MilvusMetadataFilter,
    MilvusMetricType,
    MilvusSearchResponse,
    PreparedSearchParams,
    QueryEmbedding,
)
from crewai.rag.types import BaseRecord, SearchResult


METADATA_FIELD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RESERVED_METADATA_FIELDS = {"id", "vector", "content", "metadata"}
MILVUS_LITE_COSINE_DISTANCE_VERSIONS = {"3.0", "3.0.0"}


def _ensure_list_embedding(embedding: QueryEmbedding | Any) -> list[float]:
    """Convert embedding to list[float] format if needed."""
    if isinstance(embedding, list):
        return [float(value) for value in embedding]

    if hasattr(embedding, "tolist"):
        value = embedding.tolist()
        if isinstance(value, list):
            return [float(item) for item in value]

    raise TypeError("Embedding function must return a list of floats")


def _is_sync_client(client: object) -> bool:
    """Type guard to check if the client is a synchronous MilvusClient."""
    return isinstance(client, SyncMilvusClient)


def _is_async_embedding_function(
    func: EmbeddingFunction | AsyncEmbeddingFunction,
) -> TypeGuard[AsyncEmbeddingFunction]:
    """Type guard to check if the embedding function is async."""
    return asyncio.iscoroutinefunction(func)


def _validate_metric_type(metric_type: str) -> MilvusMetricType:
    """Validate and normalize a Milvus metric type."""
    normalized = metric_type.upper()
    if normalized not in VALID_METRIC_TYPES:
        raise ValueError(
            f"Unsupported Milvus metric type '{metric_type}'. "
            f"Expected one of {sorted(VALID_METRIC_TYPES)}"
        )
    return cast(MilvusMetricType, normalized)


def _create_schema(client: Any, dimension: int) -> Any:
    """Create the explicit Milvus schema for CrewAI RAG documents."""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=DEFAULT_ID_MAX_LENGTH,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=DEFAULT_CONTENT_MAX_LENGTH,
    )
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    return schema


def _create_index_params(client: Any, metric_type: str) -> Any:
    """Create AUTOINDEX parameters for Milvus collection fields."""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type=metric_type,
    )
    return index_params


def _create_collection(
    client: Any,
    collection_name: str,
    dimension: int,
    metric_type: str,
    consistency_level: str | None,
) -> None:
    """Create a Milvus collection with the provider schema and index."""
    schema = _create_schema(client=client, dimension=dimension)
    index_params = _create_index_params(client=client, metric_type=metric_type)
    create_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "schema": schema,
        "index_params": index_params,
    }
    if consistency_level is not None:
        create_kwargs["consistency_level"] = consistency_level

    client.create_collection(**create_kwargs)


def _field_matches_type(field: dict[str, Any], expected: Any) -> bool:
    """Return whether a described Milvus field matches the expected DataType."""
    value = field.get("type")
    if value == expected or value == expected.value:
        return True
    if getattr(value, "name", None) == expected.name:
        return True
    return str(value) in {expected.name, str(expected.value)}


def _field_param_int(field: dict[str, Any], key: str) -> int | None:
    """Extract an integer field parameter from a Milvus field description."""
    params = field.get("params")
    if not isinstance(params, dict) or key not in params:
        return None
    value = params[key]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_collection_schema(
    client: Any,
    collection_name: str,
    dimension: int,
) -> None:
    """Validate an existing Milvus collection before reusing it."""
    description = client.describe_collection(collection_name=collection_name)
    fields = {
        field.get("name"): field
        for field in description.get("fields", [])
        if isinstance(field, dict)
    }

    id_field = fields.get("id")
    vector_field = fields.get("vector")
    content_field = fields.get("content")
    metadata_field = fields.get("metadata")

    if not id_field or not id_field.get("is_primary"):
        raise ValueError(f"Collection '{collection_name}' must have primary field 'id'")
    if not _field_matches_type(id_field, DataType.VARCHAR):
        raise ValueError(f"Collection '{collection_name}' field 'id' must be VARCHAR")
    if _field_param_int(id_field, "max_length") != DEFAULT_ID_MAX_LENGTH:
        raise ValueError(
            f"Collection '{collection_name}' field 'id' must use max_length "
            f"{DEFAULT_ID_MAX_LENGTH}"
        )

    if not vector_field or not _field_matches_type(vector_field, DataType.FLOAT_VECTOR):
        raise ValueError(
            f"Collection '{collection_name}' must have FLOAT_VECTOR field 'vector'"
        )
    if _field_param_int(vector_field, "dim") != dimension:
        raise ValueError(
            f"Collection '{collection_name}' vector dimension does not match "
            f"configured dimension {dimension}"
        )

    if not content_field or not _field_matches_type(content_field, DataType.VARCHAR):
        raise ValueError(
            f"Collection '{collection_name}' must have VARCHAR field 'content'"
        )
    if _field_param_int(content_field, "max_length") != DEFAULT_CONTENT_MAX_LENGTH:
        raise ValueError(
            f"Collection '{collection_name}' field 'content' must use max_length "
            f"{DEFAULT_CONTENT_MAX_LENGTH}"
        )

    if not metadata_field or not _field_matches_type(metadata_field, DataType.JSON):
        raise ValueError(
            f"Collection '{collection_name}' must have JSON field 'metadata'"
        )


def _ensure_collection(
    client: Any,
    collection_name: str,
    dimension: int,
    metric_type: str,
    consistency_level: str | None,
) -> None:
    """Create a collection if needed, otherwise validate its schema."""
    if client.has_collection(collection_name=collection_name):
        _validate_collection_schema(
            client=client,
            collection_name=collection_name,
            dimension=dimension,
        )
        return

    _create_collection(
        client=client,
        collection_name=collection_name,
        dimension=dimension,
        metric_type=metric_type,
        consistency_level=consistency_level,
    )


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    """Normalize BaseRecord metadata to a dictionary."""
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    return dict(metadata)


def _document_id(doc: BaseRecord, metadata: dict[str, Any]) -> str:
    """Return a stable document ID for a BaseRecord."""
    if "doc_id" in doc:
        return str(doc["doc_id"])
    if "doc_id" in metadata:
        return str(metadata["doc_id"])

    content_for_hash = doc["content"]
    if metadata:
        metadata_str = json.dumps(metadata, sort_keys=True)
        content_for_hash = f"{content_for_hash}|{metadata_str}"
    return hashlib.sha256(content_for_hash.encode()).hexdigest()


def _prepare_documents_for_milvus(
    documents: list[BaseRecord],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    """Prepare CrewAI documents for Milvus upsert."""
    if len(documents) != len(embeddings):
        raise ValueError("Documents and embeddings must have the same length")

    rows_by_id: dict[str, dict[str, Any]] = {}
    for doc, embedding in zip(documents, embeddings, strict=True):
        metadata = _normalize_metadata(doc.get("metadata"))
        doc_id = _document_id(doc=doc, metadata=metadata)
        rows_by_id[doc_id] = {
            "id": doc_id,
            "vector": embedding,
            "content": doc["content"],
            "metadata": metadata,
        }

    return list(rows_by_id.values())


def _validate_metadata_field_name(field_name: str) -> None:
    """Validate a metadata key before using it in a Milvus expression."""
    if not METADATA_FIELD_PATTERN.fullmatch(field_name):
        raise ValueError(
            "Milvus metadata filter keys must start with a letter or underscore "
            "and contain only letters, numbers, and underscores"
        )
    if field_name in RESERVED_METADATA_FIELDS:
        raise ValueError(
            f"Milvus metadata filter key '{field_name}' conflicts with a "
            "reserved collection field"
        )


def _format_filter_value(value: Any) -> str:
    """Format a supported scalar value for a Milvus filter expression."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Milvus metadata filter float values must be finite")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)

    raise ValueError(
        "Milvus metadata filters support only string, integer, float, and "
        "boolean values"
    )


def _build_filter_expression(
    metadata_filter: MilvusMetadataFilter | None,
) -> str | None:
    """Build a safe Milvus filter expression from metadata equality filters."""
    if not metadata_filter:
        return None

    expressions: list[str] = []
    for key, value in metadata_filter.items():
        _validate_metadata_field_name(key)
        formatted_value = _format_filter_value(value)
        expressions.append(f'metadata["{key}"] == {formatted_value}')

    return " and ".join(expressions)


def _prepare_search_params(
    collection_name: str,
    query_embedding: QueryEmbedding,
    limit: int,
    metric_type: str,
    metadata_filter: MilvusMetadataFilter | None,
) -> PreparedSearchParams:
    """Prepare parameters for MilvusClient.search."""
    search_kwargs: PreparedSearchParams = {
        "collection_name": collection_name,
        "data": [_ensure_list_embedding(query_embedding)],
        "limit": limit,
        "output_fields": ["id", "content", "metadata"],
        "search_params": {"metric_type": metric_type},
    }

    filter_expression = _build_filter_expression(metadata_filter)
    if filter_expression:
        search_kwargs["filter"] = filter_expression

    return search_kwargs


@cache
def _milvus_lite_uses_cosine_distance() -> bool:
    """Return whether the installed Milvus Lite version reports cosine distance."""
    try:
        version = importlib.metadata.version("milvus-lite")
    except importlib.metadata.PackageNotFoundError:
        return False
    base_version = version.split("+", maxsplit=1)[0]
    return base_version in MILVUS_LITE_COSINE_DISTANCE_VERSIONS


def _normalize_milvus_score(raw_score: float, metric_type: str) -> float:
    """Normalize a Milvus raw score or distance to CrewAI's score contract."""
    normalized_metric = _validate_metric_type(metric_type)
    if normalized_metric == "COSINE":
        if _milvus_lite_uses_cosine_distance():
            score = 1.0 - 0.5 * raw_score
        else:
            score = (raw_score + 1.0) / 2.0
    elif normalized_metric == "IP":
        # IP is already a higher-is-better Milvus score and may be unbounded.
        return raw_score
    elif normalized_metric == "L2":
        score = 1.0 / (1.0 + raw_score)
    else:
        raise ValueError(f"Unsupported Milvus metric type: {metric_type}")

    return max(0.0, min(1.0, score))


def _extract_hit_entity(hit: dict[str, Any]) -> dict[str, Any]:
    """Extract the entity payload from a Milvus search hit."""
    entity = hit.get("entity")
    return entity if isinstance(entity, dict) else {}


def _extract_hit_score(hit: dict[str, Any]) -> float:
    """Extract a raw score from a Milvus search hit."""
    value = hit.get("distance", hit.get("score", 0.0))
    return float(value)


def _process_search_results(
    response: MilvusSearchResponse,
    metric_type: str,
    score_threshold: float | None,
) -> list[SearchResult]:
    """Convert Milvus search response into CrewAI SearchResult dictionaries."""
    results: list[SearchResult] = []
    hits = response[0] if response else []

    for hit in hits:
        entity = _extract_hit_entity(hit)
        score = _normalize_milvus_score(
            raw_score=_extract_hit_score(hit),
            metric_type=metric_type,
        )
        if score_threshold is not None and score < score_threshold:
            continue

        metadata = entity.get("metadata")
        result: SearchResult = {
            "id": str(hit.get("id", entity.get("id", ""))),
            "content": str(entity.get("content", "")),
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            "score": score,
        }
        results.append(result)

    results.sort(key=lambda result: result["score"], reverse=True)
    return results
