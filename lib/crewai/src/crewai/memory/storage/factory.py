"""Pluggable default storage backend for the unified memory system.

By default, :class:`~crewai.memory.unified_memory.Memory` builds a built-in
vector store from its ``storage`` spec string (LanceDB, or Qdrant for the
``"qdrant-edge"`` spec). Registering a factory via
:func:`set_memory_storage_factory` lets an application route memory through a
custom :class:`~crewai.memory.storage.backend.StorageBackend` -- a different
vector store, a remote service, an in-memory fake for tests -- without
subclassing ``Memory`` or threading an explicit ``storage=`` instance through
every construction site.

This mirrors :func:`crewai_core.lock_store.set_lock_backend`: a one-time,
process-wide setter intended for application startup. Pass ``None`` to restore
the built-in default.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from crewai.memory.storage.backend import StorageBackend

# Receives the raw ``storage`` spec string and returns a backend to use, or
# ``None`` to defer to the built-in selection for that spec.
MemoryStorageFactory = Callable[[str], "StorageBackend | None"]

_factory: MemoryStorageFactory | None = None


def set_memory_storage_factory(factory: MemoryStorageFactory | None) -> None:
    """Replace the process-wide default memory storage factory.

    Intended for one-time setup at startup. Pass ``None`` to restore the
    built-in LanceDB/Qdrant selection. Only affects ``Memory`` instances
    constructed afterwards; an explicit ``storage=`` instance always wins.

    The factory is consulted for every string ``storage`` spec, so it must
    return ``None`` for specs it does not handle to let the built-in
    LanceDB/Qdrant/path selection take over.
    """
    global _factory
    _factory = factory


def resolve_memory_storage(spec: str, config: Optional[dict] = None) -> StorageBackend | None:
    """Return the registered factory's backend for ``spec``, or ``None``.

    ``None`` means no factory is registered or it declined this spec; the
    caller then falls back to the built-in selection.
    """
    # First, respect user-registered custom factories if available
    factory = _factory
    if factory is not None:
        custom_backend = factory(spec)
        if custom_backend is not None:
            return custom_backend
            
    # Built-in fallback to Mimir storage if the spec matches
    if spec == "mimir":
        from crewai.memory.storage.mimir_storage import MimirStorage
        return MimirStorage(config=config)

    return None
