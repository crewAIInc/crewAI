"""Tests for the pluggable memory storage factory seam.

We verify our own logic: the set/get round-trip, that a registered factory is
consulted for string ``storage`` specs (and receives the spec), and that an
explicit ``storage=`` instance bypasses the factory entirely.
"""

from __future__ import annotations

import pytest

import crewai.memory.storage.factory as factory
from crewai.memory.unified_memory import Memory


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset the factory around each test without clobbering preexisting state."""
    original = factory._factory
    factory.set_memory_storage_factory(None)
    yield
    factory.set_memory_storage_factory(original)


def test_resolve_reflects_registered_factory():
    sentinel = object()
    assert factory.resolve_memory_storage("lancedb") is None

    factory.set_memory_storage_factory(lambda spec: sentinel)
    assert factory.resolve_memory_storage("lancedb") is sentinel

    factory.set_memory_storage_factory(None)
    assert factory.resolve_memory_storage("lancedb") is None


def test_factory_backend_used_for_string_spec():
    sentinel = object()
    factory.set_memory_storage_factory(lambda spec: sentinel)

    mem = Memory(storage="lancedb")

    assert mem._storage is sentinel


def test_factory_receives_the_raw_spec():
    seen: list[str] = []

    def make(spec):
        seen.append(spec)
        return object()

    factory.set_memory_storage_factory(make)
    Memory(storage="some/custom/path")

    assert seen == ["some/custom/path"]


def test_explicit_storage_instance_bypasses_factory():
    factory_called = False

    def make(spec):
        nonlocal factory_called
        factory_called = True
        return object()

    factory.set_memory_storage_factory(make)

    explicit = object()
    mem = Memory(storage=explicit)  # type: ignore[arg-type]

    assert mem._storage is explicit
    assert factory_called is False
