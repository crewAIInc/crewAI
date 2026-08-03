"""Tests for the RAG client factory registry seam.

We verify our own logic: a registered factory is used for its provider,
factories override the built-in providers, unregister removes them, and an
unknown provider still raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import crewai.rag.factory as factory


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry around each test without clobbering preexisting state."""
    original = dict(factory._factories)
    factory._factories.clear()
    yield
    factory._factories.clear()
    factory._factories.update(original)


def test_registered_factory_is_used_for_its_provider():
    sentinel = object()
    factory.register_rag_client_factory("custom", lambda config: sentinel)

    assert factory.create_client(SimpleNamespace(provider="custom")) is sentinel


def test_factory_receives_the_config():
    seen: list[object] = []
    config = SimpleNamespace(provider="custom")
    factory.register_rag_client_factory("custom", lambda cfg: seen.append(cfg) or object())

    factory.create_client(config)

    assert seen == [config]


def test_factory_overrides_builtin_provider():
    sentinel = object()
    factory.register_rag_client_factory("chromadb", lambda config: sentinel)

    # Resolves via the registry without importing the built-in chromadb factory.
    assert factory.create_client(SimpleNamespace(provider="chromadb")) is sentinel


def test_unregister_removes_factory():
    factory.register_rag_client_factory("custom", lambda config: object())
    factory.unregister_rag_client_factory("custom")

    with pytest.raises(ValueError, match="Unsupported provider: custom"):
        factory.create_client(SimpleNamespace(provider="custom"))


def test_unregister_unknown_provider_is_noop():
    factory.unregister_rag_client_factory("never-registered")


def test_unknown_provider_still_raises():
    with pytest.raises(ValueError, match="Unsupported provider: nope"):
        factory.create_client(SimpleNamespace(provider="nope"))
