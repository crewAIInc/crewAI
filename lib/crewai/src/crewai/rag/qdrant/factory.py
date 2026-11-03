"""Factory functions for creating Qdrant clients from configuration."""

from collections.abc import Callable
import functools
import json
import os
import threading
from typing import Any

from qdrant_client import QdrantClient as SyncQdrantClientBase

from crewai.rag.qdrant.client import QdrantClient
from crewai.rag.qdrant.config import QdrantConfig


class _LockedQdrantClient(SyncQdrantClientBase):
    """Qdrant client whose public methods run one at a time.

    Local (on-disk) Qdrant keeps a single SQLite-backed store, which is not safe
    for concurrent use from several threads: simultaneous writes (or a read during
    a write) raise ``InterfaceError`` and lose points. Clients that share a local
    path also share this one object, so every call goes through one lock.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Set before the base class runs, which may already call public methods.
        self._call_lock = threading.RLock()
        super().__init__(**kwargs)

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if name.startswith("_") or not callable(attr):
            return attr
        lock = super().__getattribute__("_call_lock")
        return _with_lock(lock, attr)


def _with_lock(lock: threading.RLock, method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def locked(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return method(*args, **kwargs)

    return locked


# Qdrant's local (on-disk) mode locks its storage folder, so a second client on
# the same path fails. Clients that share a path therefore share one underlying
# Qdrant client; each CrewAI client still has its own embedding function.
_local_clients: dict[str, SyncQdrantClientBase] = {}
_local_clients_lock = threading.Lock()


def _get_qdrant_client(config: QdrantConfig) -> SyncQdrantClientBase:
    """Return the Qdrant client for the config, reusing local clients per path.

    The path is normalized (``~`` expanded, symlinks and relative parts resolved)
    so different spellings of the same folder share one client.
    """
    path = config.options.get("path")
    if path is None:
        return SyncQdrantClientBase(**config.options)

    normalized = os.path.realpath(os.path.expanduser(os.fspath(path)))
    with _local_clients_lock:
        if normalized not in _local_clients:
            try:
                # Every call runs under the client's lock, so using the SQLite
                # connection from other threads is safe. Without this flag,
                # builds where SQLite is not THREADSAFE=1 reject those threads.
                _local_clients[normalized] = _LockedQdrantClient(
                    **{
                        **config.options,
                        "path": normalized,
                        "force_disable_check_same_thread": True,
                    }
                )
            except RuntimeError as e:
                if "already accessed by another instance" not in str(e):
                    raise
                raise RuntimeError(
                    f"Local Qdrant storage at '{normalized}' is already in use by "
                    "another process. Local Qdrant allows one process per storage "
                    "folder (this can happen with multiple workers or replicas, or "
                    "when another script, notebook or server is running in the same "
                    "project). Use a Qdrant server instead, for example "
                    "QdrantConfig(options={'url': 'http://localhost:6333'}), or keep "
                    "ChromaDB with set_rag_config(ChromaDBConfig())."
                ) from e
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Could not read the local Qdrant storage at '{normalized}'. "
                    "This happens when another process is creating the same folder "
                    "at the same time (local Qdrant allows one process per storage "
                    "folder), or when its meta.json is corrupted. Use a Qdrant "
                    "server instead, for example "
                    "QdrantConfig(options={'url': 'http://localhost:6333'}), or keep "
                    "ChromaDB with set_rag_config(ChromaDBConfig())."
                ) from e
        return _local_clients[normalized]


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
