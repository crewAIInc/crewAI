"""Factory functions for creating ChromaDB clients."""

from typing import cast

from chromadb import Client
from chromadb.api.types import Embeddable, EmbeddingFunction as ChromaEmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.rag.chromadb.client import ChromaDBClient


def create_client(config: ChromaDBConfig) -> ChromaDBClient:
    """Create a ChromaDBClient from configuration.

    Args:
        config: ChromaDB configuration object.

    Returns:
        Configured ChromaDBClient instance.
    """
    chromadb_client = Client(
        settings=config.settings, tenant=config.tenant, database=config.database
    )

    client = ChromaDBClient()
    client.client = chromadb_client
    client.embedding_function = cast(
        ChromaEmbeddingFunction[Embeddable], DefaultEmbeddingFunction()
    )

    return client
