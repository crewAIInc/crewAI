"""Type definitions specific to Qdrant implementation."""

from typing import Any, TypedDict
from typing_extensions import NotRequired

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient as SyncQdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    HnswConfigDiff,
    InitFrom,
    IsEmptyCondition,
    IsNullCondition,
    NestedCondition,
    OptimizersConfigDiff,
    ProductQuantization,
    ScalarQuantization,
    ShardingMethod,
    SparseVectorParams,
    VectorParams,
    WalConfigDiff,
)

from crewai.rag.core.base_client import BaseCollectionParams

QdrantClientType = SyncQdrantClient | AsyncQdrantClient

QueryEmbedding = list[float] | np.ndarray[Any, np.dtype[np.floating[Any]]]

BasicConditions = FieldCondition | IsEmptyCondition | IsNullCondition
StructuralConditions = HasIdCondition | HasVectorCondition | NestedCondition
FilterCondition = BasicConditions | StructuralConditions | Filter

MetadataFilterValue = bool | int | str
MetadataFilter = dict[str, MetadataFilterValue]

QuantizationConfig = ScalarQuantization | ProductQuantization | BinaryQuantization

InitFromType = InitFrom | str


class QdrantCollectionCreateParams(BaseCollectionParams, total=False):
    """Parameters for creating a Qdrant collection.

    This class extends BaseCollectionParams to include any additional
    parameters specific to Qdrant collection creation.
    """

    vectors_config: VectorParams | dict[str, VectorParams]
    sparse_vectors_config: dict[str, SparseVectorParams]
    shard_number: int  # Number of shards
    sharding_method: ShardingMethod
    replication_factor: int  # Number of replicas
    write_consistency_factor: int  # Write consistency factor
    on_disk_payload: bool  # Store payload on disk
    hnsw_config: HnswConfigDiff
    optimizers_config: OptimizersConfigDiff
    wal_config: WalConfigDiff
    quantization_config: QuantizationConfig
    init_from: InitFromType
    timeout: int


class PreparedSearchParams(TypedDict):
    """Type definition for prepared Qdrant search parameters."""

    collection_name: str
    query: list[float]
    limit: int
    with_payload: bool
    with_vectors: bool
    score_threshold: NotRequired[float]
    query_filter: NotRequired[Filter]
