"""Factory functions for creating ChromaDB clients."""

import os
from hashlib import md5

import portalocker
from chromadb import PersistentClient

from crewai.rag.chromadb.client import ChromaDBClient
from crewai.rag.chromadb.config import ChromaDBConfig


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
    lock_id = md5(persist_dir.encode(), usedforsecurity=False).hexdigest()
    lockfile = os.path.join(persist_dir, f"chromadb-{lock_id}.lock")

    with portalocker.Lock(lockfile):
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
    )
