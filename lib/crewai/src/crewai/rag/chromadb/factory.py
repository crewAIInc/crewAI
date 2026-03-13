"""Factory functions for creating ChromaDB clients."""

import os

from chromadb import PersistentClient

from crewai.rag.chromadb.client import ChromaDBClient
from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.utilities.lock_store import lock


def create_client(config: ChromaDBConfig) -> ChromaDBClient:
    """Create a ChromaDBClient from configuration.

    Args:
        config: ChromaDB configuration object.

    Returns:
        Configured ChromaDBClient instance.

    Notes:
        Need to update to use chromadb.Client to support more client types in the near future.
    """

    persist_dir = config.settings.persist_directory
    os.makedirs(persist_dir, exist_ok=True)

    with lock(f"chromadb:{persist_dir}"):
        client = PersistentClient(
            path=persist_dir,
            settings=config.settings,
            tenant=config.tenant,
            database=config.database,
        )

    return ChromaDBClient(
        client=client,
        embedding_function=config.embedding_function,
        default_limit=config.limit,
        default_score_threshold=config.score_threshold,
        default_batch_size=config.batch_size,
        lock_name=f"chromadb:{persist_dir}",
    )
