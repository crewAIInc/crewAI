"""Constants for Elasticsearch RAG implementation."""

from typing import Final

DEFAULT_HOST: Final[str] = "localhost"
DEFAULT_PORT: Final[int] = 9200
DEFAULT_INDEX_SETTINGS: Final[dict] = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
}
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_DIMENSION: Final[int] = 384
