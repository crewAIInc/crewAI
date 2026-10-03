"""How a deployment swaps in a different store.

Kept as a process-wide factory rather than a tool argument on purpose: the
crews that need this are already written and deployed, and the point is that
they keep working unchanged when the runtime is ephemeral. An integration
package registers its factory at import time, and every file tool constructed
afterwards picks it up.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import threading

from crewai_tools.file_storage.base import FileStore
from crewai_tools.file_storage.local import LocalFileStore


logger = logging.getLogger(__name__)

#: A factory returns the store to use, or ``None`` to decline — which lets an
#: integration arm itself only when its backing service is actually
#: configured, and fall back to the local filesystem everywhere else.
FileStoreFactory = Callable[[], FileStore | None]

_lock = threading.Lock()
_factory: FileStoreFactory | None = None
_local = LocalFileStore()


def register_file_store_factory(factory: FileStoreFactory | None) -> None:
    """Install the factory consulted for every new file tool.

    Args:
        factory: Callable returning a :class:`FileStore`, or ``None`` to
            decline and leave the local filesystem in place. Passing
            ``None`` as the factory itself unregisters.
    """
    global _factory
    with _lock:
        _factory = factory


def reset_file_store_factory() -> None:
    """Drop any registered factory. Intended for tests."""
    register_file_store_factory(None)


def resolve_file_store() -> FileStore:
    """Return the store the file tools should use.

    A factory that raises is not allowed to take the tools down with it: an
    integration failing to initialize should degrade to the local filesystem
    — the behavior before it was installed — not break file I/O outright.
    """
    with _lock:
        factory = _factory

    if factory is None:
        return _local

    try:
        store = factory()
    except Exception:
        logger.warning(
            "file store factory raised; falling back to the local filesystem",
            exc_info=True,
        )
        return _local

    if store is None:
        return _local
    return store
