"""Constants for RAG configuration."""

from typing import Final


DISCRIMINATOR: Final[str] = "provider"

DEFAULT_RAG_PROVIDER: Final[str] = "qdrant"
DEFAULT_RAG_CONFIG_PATH: Final[str] = "crewai.rag.qdrant.config"
DEFAULT_RAG_CONFIG_CLASS: Final[str] = "QdrantConfig"
