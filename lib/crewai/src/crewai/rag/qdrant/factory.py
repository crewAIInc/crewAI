"""Factory functions for creating Qdrant clients from configuration."""

import threading

from qdrant_client import QdrantClient as SyncQdrantClientBase

from crewai.rag.qdrant.client import QdrantClient
from crewai.rag.qdrant.config import QdrantConfig


# Qdrant's local (on-disk) mode locks its storage folder, so a second client on
# the same path fails. Clients that share a path therefore share one underlying
# Qdrant client; each CrewAI client still has its own embedding function.
_local_clients: dict[str, SyncQdrantClientBase] = {}
_local_clients_lock = threading.Lock()


def _get_qdrant_client(config: QdrantConfig) -> SyncQdrantClientBase:
    """Return the Qdrant client for the config, reusing local clients per path."""
    path = config.options.get("path")
    if path is None:
        return SyncQdrantClientBase(**config.options)

    with _local_clients_lock:
        if path not in _local_clients:
            _local_clients[path] = SyncQdrantClientBase(**config.options)
        return _local_clients[path]


def create_client(config: QdrantConfig) -> QdrantClient:
    """Create a Qdrant client from configuration.

    Args:
        config: The Qdrant configuration.

    Returns:
        A configured QdrantClient instance.
    """

    return QdrantClient(
        client=_get_qdrant_client(config),
        embedding_function=config.embedding_function,
        default_limit=config.limit,
        default_score_threshold=config.score_threshold,
        default_batch_size=config.batch_size,
    )
