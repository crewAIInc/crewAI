"""Constants for Milvus RAG provider."""

from typing import Final


DEFAULT_URI: Final[str] = "./milvus.db"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
DEFAULT_DIMENSION: Final[int] = 1536
DEFAULT_METRIC_TYPE: Final[str] = "COSINE"
DEFAULT_ID_MAX_LENGTH: Final[int] = 512
DEFAULT_CONTENT_MAX_LENGTH: Final[int] = 65535
VALID_CONSISTENCY_LEVELS: Final[set[str]] = {
    "Strong",
    "Session",
    "Bounded",
    "Eventually",
}
VALID_METRIC_TYPES: Final[set[str]] = {"COSINE", "IP", "L2"}
