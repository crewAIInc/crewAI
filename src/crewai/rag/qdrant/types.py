"""Type definitions specific to Qdrant implementation."""

from typing import Annotated, Any, TypedDict
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

QueryEmbedding = Annotated[
    list[float] | np.ndarray[Any, np.dtype[np.floating[Any]]],
    "Embedding vector as list of floats or numpy array",
]

BasicConditions = FieldCondition | IsEmptyCondition | IsNullCondition
StructuralConditions = HasIdCondition | HasVectorCondition | NestedCondition
FilterCondition = BasicConditions | StructuralConditions | Filter

MetadataFilterValue = bool | int | str
MetadataFilter = dict[str, MetadataFilterValue]


class QdrantCollectionCreateParams(BaseCollectionParams, total=False):
    """Parameters for creating a Qdrant collection.

    This class extends BaseCollectionParams to include any additional
    parameters specific to Qdrant collection creation.
    """

    vectors_config: VectorParams | dict[str, VectorParams]
    sparse_vectors_config: dict[str, SparseVectorParams]
    shard_number: Annotated[int, "Number of shards (default: 1)"]
    sharding_method: ShardingMethod
    replication_factor: Annotated[int, "Number of replicas per shard (default: 1)"]
    write_consistency_factor: Annotated[int, "Await N replicas on write (default: 1)"]
    on_disk_payload: Annotated[bool, "Store payload on disk instead of RAM"]
    hnsw_config: HnswConfigDiff
    optimizers_config: OptimizersConfigDiff
    wal_config: WalConfigDiff
    quantization_config: ScalarQuantization | ProductQuantization | BinaryQuantization
    init_from: InitFrom | str
    timeout: Annotated[int, "Operation timeout in seconds"]


class PreparedSearchParams(TypedDict):
    """Type definition for prepared Qdrant search parameters."""

    collection_name: str
    query: list[float]
    limit: Annotated[int, "Max results to return"]
    with_payload: Annotated[bool, "Include payload in results"]
    with_vectors: Annotated[bool, "Include vectors in results"]
    score_threshold: NotRequired[Annotated[float, "Min similarity score (0-1)"]]
    query_filter: NotRequired[Filter]
