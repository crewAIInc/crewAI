"""Factory functions for creating RAG clients from configuration."""

import importlib
from typing import cast

from crewai.rag.config.optional_imports.protocols import (
    ChromaFactoryModule,
    QdrantFactoryModule,
)
from crewai.rag.config.types import RagConfigType
from crewai.rag.core.base_client import BaseClient


def _import_rag_factory(module_path: str, purpose: str) -> object:
    """Import an optional RAG factory module with a clear install hint."""
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        package_name = module_path.split(".")[0]
        raise ImportError(
            f"{purpose} requires the optional dependency '{module_path}'.\n"
            f"Install it with: uv add {package_name}"
        ) from exc


def create_client(config: RagConfigType) -> BaseClient:
    """Create a client from configuration using the appropriate factory.

    Args:
        config: The RAG client configuration.

    Returns:
        The created client instance.

    Raises:
        ValueError: If the configuration provider is not supported.
    """

    if config.provider == "chromadb":
        chromadb_mod = cast(
            ChromaFactoryModule,
            _import_rag_factory(
                "crewai.rag.chromadb.factory",
                purpose="The 'chromadb' provider",
            ),
        )
        return chromadb_mod.create_client(config)

    if config.provider == "qdrant":
        qdrant_mod = cast(
            QdrantFactoryModule,
            _import_rag_factory(
                "crewai.rag.qdrant.factory",
                purpose="The 'qdrant' provider",
            ),
        )
        return qdrant_mod.create_client(config)

    raise ValueError(f"Unsupported provider: {config.provider}")
