"""Factory functions for creating RAG clients from configuration."""

from crewai.rag.core.base_client import BaseClient
from crewai.rag.config.types import RagConfigType


def create_client(config: RagConfigType) -> BaseClient:
    """Create a client from configuration using the appropriate factory.

    Args:
        config: The RAG client configuration.

    Returns:
        The created client instance.

    Raises:
        ValueError: If the configuration provider is not supported.
    """
    if config.provider == "chromadb":
        from crewai.rag.chromadb.factory import create_client as create_chromadb_client

        return create_chromadb_client(config)

    raise ValueError(f"Unsupported RAG provider: {config.provider}")
