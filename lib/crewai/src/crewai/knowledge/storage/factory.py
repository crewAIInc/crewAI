"""Pluggable default storage backend for knowledge collections.

By default, :class:`~crewai.knowledge.knowledge.Knowledge` builds a
:class:`~crewai.knowledge.storage.knowledge_storage.KnowledgeStorage` when no
explicit ``storage=`` is given. Registering a factory via
:func:`set_knowledge_storage_factory` lets an application back knowledge with a
custom :class:`~crewai.knowledge.storage.base_knowledge_storage.BaseKnowledgeStorage`
without subclassing ``Knowledge`` or passing a ``storage=`` instance at every
call site.

This mirrors :func:`crewai_core.lock_store.set_lock_backend`: a one-time,
process-wide setter intended for application startup. Pass ``None`` to restore
the built-in default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
    from crewai.rag.embeddings.types import EmbedderConfig

# Receives the same inputs as the built-in default -- the embedder config and
# collection name -- and returns a storage backend, or ``None`` to defer to the
# built-in ``KnowledgeStorage``.
KnowledgeStorageFactory = Callable[
    ["EmbedderConfig | None", "str | None"], "BaseKnowledgeStorage | None"
]

_factory: KnowledgeStorageFactory | None = None


def set_knowledge_storage_factory(factory: KnowledgeStorageFactory | None) -> None:
    """Replace the process-wide default knowledge storage factory.

    Intended for one-time setup at startup. Pass ``None`` to restore the
    built-in ``KnowledgeStorage``. Only affects ``Knowledge`` instances
    constructed afterwards; an explicit ``storage=`` instance always wins.
    """
    global _factory
    _factory = factory


def resolve_knowledge_storage(
    embedder: EmbedderConfig | None, collection_name: str | None
) -> BaseKnowledgeStorage | None:
    """Return the registered factory's backend, or ``None`` for the built-in.

    ``None`` means no factory is registered or it declined; the caller then
    falls back to the built-in ``KnowledgeStorage``.
    """
    factory = _factory
    return factory(embedder, collection_name) if factory is not None else None
