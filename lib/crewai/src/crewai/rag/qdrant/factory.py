"""Factory functions for creating Qdrant clients from configuration."""

import atexit
from collections.abc import Callable
import contextlib
import functools
import json
import logging
import os
import shutil
import threading
import time
from typing import Any

from crewai_core.paths import db_storage_path
from qdrant_client import QdrantClient as SyncQdrantClientBase

from crewai.rag.qdrant.client import QdrantClient
from crewai.rag.qdrant.config import QdrantConfig


logger = logging.getLogger(__name__)


class _LockedQdrantClient(SyncQdrantClientBase):
    """Qdrant client whose public methods run one at a time.

    Local (on-disk) Qdrant keeps a single SQLite-backed store, which is not safe
    for concurrent use from several threads: simultaneous writes (or a read during
    a write) raise ``InterfaceError`` and lose points. Clients that share a local
    path also share this one object, so every call goes through one lock.

    The store belongs to the process that opened it. A client inherited through
    ``os.fork()`` still points at the parent's folder and SQLite connection, so
    any call from another process raises instead of touching the parent's data.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Set before the base class runs, which may already call public methods.
        self._call_lock = threading.RLock()
        self._owner_pid = os.getpid()
        super().__init__(**kwargs)

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if name.startswith("_") or not callable(attr):
            return attr
        owner_pid = super().__getattribute__("_owner_pid")
        if os.getpid() != owner_pid:
            if name == "close":
                # Closing here would release the parent's folder lock.
                return lambda: None
            raise RuntimeError(
                f"This local Qdrant client was opened by process {owner_pid} and "
                f"cannot be used from process {os.getpid()}. It was inherited "
                "through a fork. Create the client (or the Knowledge/Crew that "
                "uses it) inside each worker process, after the fork."
            )
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

_RMTREE_ATTEMPTS = 5
"""How often deleting the default folder is attempted before giving up.

Windows refuses to delete a file while another handle has it open, and even
after the client is closed a virus scanner or the search indexer can hold one
for a moment. A single ``rmtree`` call is therefore not enough there, while its
failure is worth retrying on every platform because the retry is free when the
first attempt succeeds.
"""

_RMTREE_RETRY_DELAY = 0.1
"""Seconds to wait between delete attempts; short, since this runs at exit."""


def _close_local_client(folder: str) -> None:
    """Drop ``folder``'s cached client and close it.

    Closing releases the folder's ``.lock`` file and every collection's
    ``storage.sqlite``. It is what lets the folder be deleted on Windows, which
    cannot unlink files that are still open. Safe to call when the client was
    never opened or was already closed.
    """
    with _local_clients_lock:
        client = _local_clients.pop(folder, None)
    if client is None:
        return
    # A close failure must not stop the folder from being removed, and it cannot
    # be reported usefully at interpreter exit.
    with contextlib.suppress(Exception):
        client.close()


def _try_remove_once(folder: str) -> bool:
    """Delete ``folder`` once.

    Returns:
        ``True`` when the folder is gone, ``False`` when a file in it was still
        open, which Windows reports as ``OSError``.
    """
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _remove_folder(folder: str) -> bool:
    """Delete ``folder``, retrying handles that are briefly still open.

    Returns:
        ``True`` when the folder is gone, ``False`` when it survived every
        attempt. Failures are never swallowed silently: the caller reports a
        survivor, because a leftover folder is what the bug report describes.
    """
    for attempt in range(_RMTREE_ATTEMPTS):
        if _try_remove_once(folder):
            return True
        if attempt < _RMTREE_ATTEMPTS - 1:
            time.sleep(_RMTREE_RETRY_DELAY)
    return False


def _remove_folder_if_owner(folder: str, owner_pid: int) -> None:
    """Delete ``folder`` at exit, but only in the process that created it.

    A forked child inherits this exit handler and must not delete its parent's
    storage when it exits.

    The folder's client is closed first: Windows cannot delete files that are
    still open, so removing the folder while the shared client held its
    ``.lock`` and ``storage.sqlite`` files left it behind on every run. A folder
    that still cannot be removed is reported rather than dropped, because a
    silent ``ignore_errors`` delete is what let the leftovers pile up unnoticed.
    """
    if os.getpid() != owner_pid:
        return
    _close_local_client(folder)
    if not _remove_folder(folder):
        logger.warning(
            "Could not remove the local Qdrant storage at '%s'. It is left in "
            "place rather than deleted while a file is still open. Remove it "
            "once the process that uses it has exited.",
            folder,
        )


def _get_qdrant_client(config: QdrantConfig) -> SyncQdrantClientBase:
    """Return the Qdrant client for the config, reusing local clients per path.

    The path is normalized (``~`` expanded, symlinks and relative parts resolved)
    so different spellings of the same folder share one client.

    Local Qdrant allows one process per folder, so the default folder gets the
    process ID added and is removed when the process exits. Several processes
    (workers, replicas, scripts) then never collide, at the cost of embedding
    documents again in each process. A folder chosen with ``path`` is used as
    given and stays one process per folder.
    """
    path = config.options.get("path")
    if path is None:
        return SyncQdrantClientBase(**config.options)

    normalized = os.path.realpath(os.path.expanduser(os.fspath(path)))
    is_default = normalized == os.path.realpath(
        os.path.join(db_storage_path(), "qdrant")
    )
    # Keyed by the folder actually opened. For the default folder that includes
    # the process ID, so a forked child opens its own folder instead of reusing
    # (and sharing the SQLite connection of) its parent's client.
    folder = f"{normalized}-pid{os.getpid()}" if is_default else normalized
    with _local_clients_lock:
        if folder not in _local_clients:
            try:
                # Every call runs under the client's lock, so using the SQLite
                # connection from other threads is safe. Without this flag,
                # builds where SQLite is not THREADSAFE=1 reject those threads.
                _local_clients[folder] = _LockedQdrantClient(
                    **{
                        **config.options,
                        "path": folder,
                        "force_disable_check_same_thread": True,
                    }
                )
            except RuntimeError as e:
                if "already accessed by another instance" not in str(e):
                    raise
                raise RuntimeError(
                    f"Local Qdrant storage at '{folder}' is already in use by "
                    "another process. Local Qdrant allows one process per storage "
                    "folder (this can happen when another script, notebook or "
                    "server uses the same path). Use a different path for each "
                    "process, a Qdrant server, for example "
                    "QdrantConfig(options={'url': 'http://localhost:6333'}), or "
                    "keep ChromaDB with set_rag_config(ChromaDBConfig())."
                ) from e
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Could not read the local Qdrant storage at '{folder}'. "
                    "This happens when another process is creating the same folder "
                    "at the same time (local Qdrant allows one process per storage "
                    "folder), or when its meta.json is corrupted. Use a Qdrant "
                    "server instead, for example "
                    "QdrantConfig(options={'url': 'http://localhost:6333'}), or keep "
                    "ChromaDB with set_rag_config(ChromaDBConfig())."
                ) from e
            if is_default:
                atexit.register(_remove_folder_if_owner, folder, os.getpid())
        return _local_clients[folder]


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