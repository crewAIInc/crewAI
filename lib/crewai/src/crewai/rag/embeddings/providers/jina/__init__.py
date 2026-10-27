"""Jina embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.jina.types import (
    JinaProviderConfig,
    JinaProviderSpec,
)


__all__ = [
    "JinaProvider",
    "JinaProviderConfig",
    "JinaProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"JinaProvider": f"{__name__}.jina_provider"})
