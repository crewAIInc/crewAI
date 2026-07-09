"""Factory functions for creating RAG clients from configuration."""

from collections.abc import Callable
from typing import cast

from crewai.rag.config.optional_imports.protocols import (
    ChromaFactoryModule,
    QdrantFactoryModule,
    TurbopufferFactoryModule,
)
from crewai.rag.config.types import RagConfigType
from crewai.rag.core.base_client import BaseClient
from crewai.utilities.import_utils import require


# RAG uses a provider-keyed registry (rather than the single-default setter
# used by the memory/knowledge/flow seams) because ``create_client`` already
# dispatches on ``config.provider`` -- the natural seam here is per-provider.
# A factory receives the RAG config and returns a client; one registered for a
# built-in provider name overrides the built-in for that provider.
RagClientFactory = Callable[[RagConfigType], BaseClient]

_factories: dict[str, RagClientFactory] = {}


def register_rag_client_factory(provider: str, factory: RagClientFactory) -> None:
    """Register a client factory for a RAG ``provider`` name.

    Lets an application plug in a client for a new provider, or override a
    built-in provider (``"chromadb"`` / ``"qdrant"``), without modifying
    :func:`create_client`. Registered factories take precedence over the
    built-ins. Intended for one-time setup at startup.
    """
    _factories[provider] = factory


def unregister_rag_client_factory(provider: str) -> None:
    """Remove a previously registered factory; a no-op if none is registered."""
    _factories.pop(provider, None)


def create_client(config: RagConfigType) -> BaseClient:
    """Create a client from configuration using the appropriate factory.

    Args:
        config: The RAG client configuration.

    Returns:
        The created client instance.

    Raises:
        ValueError: If the configuration provider is not supported.
    """

    factory = _factories.get(config.provider)
    if factory is not None:
        return factory(config)

    if config.provider == "chromadb":
        chromadb_mod = cast(
            ChromaFactoryModule,
            require(
                "crewai.rag.chromadb.factory",
                purpose="The 'chromadb' provider",
            ),
        )
        return chromadb_mod.create_client(config)

    if config.provider == "qdrant":
        qdrant_mod = cast(
            QdrantFactoryModule,
            require(
                "crewai.rag.qdrant.factory",
                purpose="The 'qdrant' provider",
            ),
        )
        return qdrant_mod.create_client(config)

    if config.provider == "turbopuffer":
        tpuf_mod = cast(
            TurbopufferFactoryModule,
            require(
                "crewai.rag.turbopuffer.factory",
                purpose="The 'turbopuffer' provider",
            ),
        )
        return tpuf_mod.create_client(config)

    raise ValueError(f"Unsupported provider: {config.provider}")
