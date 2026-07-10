"""Type definitions for the Milvus RAG provider."""

from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing_extensions import NotRequired, TypedDict

from crewai.rag.core.base_client import BaseCollectionParams


QueryEmbedding: TypeAlias = list[float]
MilvusMetricType: TypeAlias = Literal["COSINE", "IP", "L2"]
MilvusConsistencyLevel: TypeAlias = Literal[
    "Strong", "Session", "Bounded", "Eventually"
]


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions that convert text to vectors."""

    def __call__(self, text: str) -> QueryEmbedding:
        """Convert text to an embedding vector."""
        ...


class AsyncEmbeddingFunction(Protocol):
    """Protocol for async embedding functions that convert text to vectors."""

    async def __call__(self, text: str) -> QueryEmbedding:
        """Convert text to an embedding vector asynchronously."""
        ...


class MilvusEmbeddingFunctionWrapper(EmbeddingFunction):
    """Base class for Milvus embedding functions used by Pydantic validation."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for Milvus embedding functions."""
        return core_schema.any_schema()


class MilvusClientParams(TypedDict, total=False):
    """Parameters for pymilvus.MilvusClient initialization."""

    uri: str
    token: str
    db_name: str
    user: str
    password: str
    timeout: float | None


class MilvusCollectionCreateParams(BaseCollectionParams, total=False):
    """Collection creation parameters for Milvus."""

    dimension: Annotated[int, "Vector dimension for the collection"]
    metric_type: Annotated[MilvusMetricType, "Vector metric type"]
    consistency_level: MilvusConsistencyLevel


class PreparedSearchParams(TypedDict):
    """Parameters for MilvusClient.search."""

    collection_name: str
    data: list[list[float]]
    limit: int
    output_fields: list[str]
    search_params: dict[str, Any]
    filter: NotRequired[str]


MilvusMetadataValue: TypeAlias = str | int | float | bool
MilvusMetadataFilter: TypeAlias = dict[str, MilvusMetadataValue]
MilvusSearchHit: TypeAlias = dict[str, Any]
MilvusSearchResponse: TypeAlias = list[list[MilvusSearchHit]]
