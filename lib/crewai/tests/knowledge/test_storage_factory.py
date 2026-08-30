"""Tests for the pluggable knowledge storage factory seam.

We verify our own logic: the set/get round-trip, that a registered factory is
consulted when no explicit ``storage=`` is given (and receives the embedder and
collection name), and that an explicit ``storage=`` instance bypasses it.
"""

from __future__ import annotations

from typing import Any

import pytest

import crewai.knowledge.storage.factory as factory
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.rag.types import SearchResult


class _FakeKnowledgeStorage(BaseKnowledgeStorage):
    """Minimal stand-in implementing the abstract interface."""

    def search(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        return []

    async def asearch(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        return []

    def save(self, documents: list[str]) -> None:
        return None

    async def asave(self, documents: list[str]) -> None:
        return None

    def reset(self) -> None:
        return None

    async def areset(self) -> None:
        return None


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset the factory around each test without clobbering preexisting state."""
    original = factory._factory
    factory.set_knowledge_storage_factory(None)
    yield
    factory.set_knowledge_storage_factory(original)


def test_resolve_reflects_registered_factory():
    fake = _FakeKnowledgeStorage()
    assert factory.resolve_knowledge_storage(None, "docs") is None

    factory.set_knowledge_storage_factory(lambda embedder, name: fake)
    assert factory.resolve_knowledge_storage(None, "docs") is fake


def test_factory_used_when_no_explicit_storage():
    fake = _FakeKnowledgeStorage()
    factory.set_knowledge_storage_factory(lambda embedder, name: fake)

    knowledge = Knowledge(collection_name="docs", sources=[])

    assert knowledge.storage is fake


def test_factory_receives_embedder_and_collection_name():
    seen: list[tuple[object, object]] = []

    def make(embedder, collection_name):
        seen.append((embedder, collection_name))
        return _FakeKnowledgeStorage()

    factory.set_knowledge_storage_factory(make)
    Knowledge(collection_name="docs", sources=[])

    assert seen == [(None, "docs")]


def test_explicit_storage_bypasses_factory():
    factory_called = False

    def make(embedder, name):
        nonlocal factory_called
        factory_called = True
        return _FakeKnowledgeStorage()

    factory.set_knowledge_storage_factory(make)

    explicit = _FakeKnowledgeStorage()
    knowledge = Knowledge(collection_name="docs", sources=[], storage=explicit)

    assert knowledge.storage is explicit
    assert factory_called is False


def test_falsy_explicit_storage_is_honored():
    # A custom backend that is falsy (defines __bool__/__len__) must still be
    # used and operated on, not silently treated as "not initialized" by a
    # truthiness check in __init__, reset, or the source save path.
    reset_calls: list[bool] = []

    class _FalsyStorage(_FakeKnowledgeStorage):
        def __bool__(self) -> bool:
            return False

        def reset(self) -> None:
            reset_calls.append(True)

    explicit = _FalsyStorage()
    knowledge = Knowledge(collection_name="docs", sources=[], storage=explicit)

    assert knowledge.storage is explicit

    # reset must call the backend, not raise "Storage is not initialized."
    knowledge.reset()
    assert reset_calls == [True]
