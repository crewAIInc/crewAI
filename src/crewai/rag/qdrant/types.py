"""Type definitions specific to Qdrant implementation."""

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol, TypeAlias, TypedDict
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

QueryEmbedding: TypeAlias = list[float] | np.ndarray[Any, np.dtype[np.floating[Any]]]

BasicConditions = FieldCondition | IsEmptyCondition | IsNullCondition
StructuralConditions = HasIdCondition | HasVectorCondition | NestedCondition
FilterCondition = BasicConditions | StructuralConditions | Filter

MetadataFilterValue = bool | int | str
MetadataFilter = dict[str, MetadataFilterValue]


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions that convert text to vectors."""

    def __call__(self, text: str) -> QueryEmbedding:
        """Convert text to embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats or numpy array.
        """
        ...


class AsyncEmbeddingFunction(Protocol):
    """Protocol for async embedding functions that convert text to vectors."""

    async def __call__(self, text: str) -> QueryEmbedding:
        """Convert text to embedding vector asynchronously.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats or numpy array.
        """
        ...


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


class QdrantClientParams(TypedDict, total=False):
    """Parameters for QdrantClient initialization."""

    location: str | None
    url: str | None
    port: int
    grpc_port: int
    prefer_grpc: bool
    https: bool | None
    api_key: str | None
    prefix: str | None
    timeout: int | None
    host: str | None
    path: str | None
    force_disable_check_same_thread: bool
    grpc_options: dict[str, Any] | None
    auth_token_provider: Callable[[], str] | Callable[[], Awaitable[str]] | None
    cloud_inference: bool
    local_inference_batch_size: int | None
    check_compatibility: bool


class PreparedSearchParams(TypedDict):
    """Type definition for prepared Qdrant search parameters."""

    collection_name: str
    query: list[float]
    limit: Annotated[int, "Max results to return"]
    with_payload: Annotated[bool, "Include payload in results"]
    with_vectors: Annotated[bool, "Include vectors in results"]
    score_threshold: NotRequired[Annotated[float, "Min similarity score (0-1)"]]
    query_filter: NotRequired[Filter]
