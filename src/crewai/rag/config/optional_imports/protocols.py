"""Protocol definitions for RAG factory modules."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.rag.chromadb.client import ChromaDBClient
    from crewai.rag.chromadb.config import ChromaDBConfig
    from crewai.rag.qdrant.client import QdrantClient
    from crewai.rag.qdrant.config import QdrantConfig
    from crewai.rag.elasticsearch.client import ElasticsearchClient
    from crewai.rag.elasticsearch.config import ElasticsearchConfig


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


class ElasticsearchFactoryModule(Protocol):
    """Protocol for Elasticsearch factory module."""

    def create_client(self, config: ElasticsearchConfig) -> ElasticsearchClient:
        """Creates an Elasticsearch client from configuration."""
        ...
