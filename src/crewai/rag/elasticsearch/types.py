"""Type definitions for Elasticsearch RAG implementation."""

from typing import Any, Protocol, TypedDict, Union, TYPE_CHECKING
from typing_extensions import NotRequired
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

if TYPE_CHECKING:
    from typing import TypeAlias
    from elasticsearch import Elasticsearch, AsyncElasticsearch
    ElasticsearchClientType: TypeAlias = Union[Elasticsearch, AsyncElasticsearch]
else:
    try:
        from elasticsearch import Elasticsearch, AsyncElasticsearch
        ElasticsearchClientType = Union[Elasticsearch, AsyncElasticsearch]
    except ImportError:
        ElasticsearchClientType = Any


class ElasticsearchClientParams(TypedDict, total=False):
    """Parameters for Elasticsearch client initialization."""
    
    hosts: NotRequired[list[str]]
    cloud_id: NotRequired[str]
    username: NotRequired[str]
    password: NotRequired[str]
    api_key: NotRequired[str]
    use_ssl: NotRequired[bool]
    verify_certs: NotRequired[bool]
    ca_certs: NotRequired[str]
    timeout: NotRequired[int]


class ElasticsearchIndexSettings(TypedDict, total=False):
    """Settings for Elasticsearch index creation."""
    
    number_of_shards: NotRequired[int]
    number_of_replicas: NotRequired[int]
    refresh_interval: NotRequired[str]


class ElasticsearchCollectionCreateParams(TypedDict, total=False):
    """Parameters for creating Elasticsearch collections/indices."""
    
    collection_name: str
    index_settings: NotRequired[ElasticsearchIndexSettings]
    vector_dimension: NotRequired[int]
    similarity: NotRequired[str]


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions that convert text to vectors."""

    def __call__(self, text: str) -> list[float]:
        """Convert text to embedding vector.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        ...


class AsyncEmbeddingFunction(Protocol):
    """Protocol for async embedding functions that convert text to vectors."""

    async def __call__(self, text: str) -> list[float]:
        """Convert text to embedding vector asynchronously.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        ...


class ElasticsearchEmbeddingFunctionWrapper(EmbeddingFunction):
    """Base class for Elasticsearch EmbeddingFunction to work with Pydantic validation."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for Elasticsearch EmbeddingFunction.

        This allows Pydantic to handle Elasticsearch's EmbeddingFunction type
        without requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()
