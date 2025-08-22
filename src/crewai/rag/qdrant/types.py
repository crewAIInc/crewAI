"""Type definitions specific to Qdrant implementation."""

from typing import Any, TypedDict

from qdrant_client import AsyncQdrantClient, QdrantClient as SyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    NestedCondition,
)
from typing_extensions import NotRequired

from crewai.rag.core.base_client import BaseCollectionParams

QdrantClientType = SyncQdrantClient | AsyncQdrantClient

FilterCondition = (
    FieldCondition
    | IsEmptyCondition
    | IsNullCondition
    | HasIdCondition
    | HasVectorCondition
    | NestedCondition
    | Filter
)


class QdrantCollectionCreateParams(BaseCollectionParams, total=False):
    """Parameters for creating a Qdrant collection.

    This class extends BaseCollectionParams to include any additional
    parameters specific to Qdrant collection creation.
    """

    vectors_config: Any  # VectorParams or dict of VectorParams
    sparse_vectors_config: dict[str, Any]  # Sparse vector configuration
    shard_number: int  # Number of shards
    sharding_method: Any  # ShardingMethod
    replication_factor: int  # Number of replicas
    write_consistency_factor: int  # Write consistency factor
    on_disk_payload: bool  # Store payload on disk
    hnsw_config: Any  # HNSW index configuration
    optimizers_config: Any  # Optimizer settings
    wal_config: Any  # Write-ahead log configuration
    quantization_config: Any  # Quantization settings
    init_from: Any  # Initialize from another collection
    timeout: int  # Operation timeout


class PreparedSearchParams(TypedDict):
    """Type definition for prepared Qdrant search parameters."""

    collection_name: str
    query: list[float]
    limit: int
    with_payload: bool
    with_vectors: bool
    score_threshold: NotRequired[float]
    query_filter: NotRequired[Filter]
