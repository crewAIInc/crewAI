"""Type definitions for RAG tool configuration."""

from typing import Any, Literal, TypedDict

from crewai.rag.embeddings.types import ProviderSpec


class VectorDbConfig(TypedDict):
    """Configuration for vector database provider.

    Attributes:
        provider: RAG provider literal.
        config: RAG configuration options.
    """

    provider: Literal["chromadb", "qdrant"]
    config: dict[str, Any]


class RagToolConfig(TypedDict, total=False):
    """Configuration accepted by RAG tools.

    Supports embedding model and vector database configuration.

    Attributes:
        embedding_model: Embedding model configuration accepted by RAG tools.
        vectordb: Vector database configuration accepted by RAG tools.
    """

    embedding_model: ProviderSpec
    vectordb: VectorDbConfig
