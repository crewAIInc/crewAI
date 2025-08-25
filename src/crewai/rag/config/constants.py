"""Constants for RAG configuration."""

from typing import Final

DISCRIMINATOR: Final[str] = "provider"

DEFAULT_RAG_CONFIG_PATH: Final[str] = "crewai.rag.chromadb.config"
DEFAULT_RAG_CONFIG_CLASS: Final[str] = "ChromaDBConfig"
