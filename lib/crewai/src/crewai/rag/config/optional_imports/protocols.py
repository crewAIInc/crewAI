"""Protocol definitions for RAG factory modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from crewai.rag.chromadb.client import ChromaDBClient
    from crewai.rag.chromadb.config import ChromaDBConfig
    from crewai.rag.qdrant.client import QdrantClient
    from crewai.rag.qdrant.config import QdrantConfig


class ChromaFactoryModule(Protocol):
    """Protocol for ChromaDB factory module."""

    def create_client(self, config: ChromaDBConfig) -> ChromaDBClient:
        """Creates a ChromaDB client from configuration."""
        ...


class QdrantFactoryModule(Protocol):
    """Protocol for Qdrant factory module."""

    def create_client(self, config: QdrantConfig) -> QdrantClient:
        """Creates a Qdrant client from configuration."""
        ...
