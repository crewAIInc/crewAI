"""Jina embedding providers."""

from crewai.rag.embeddings.providers.jina.jina_provider import JinaProvider
from crewai.rag.embeddings.providers.jina.types import (
    JinaProviderConfig,
    JinaProviderSpec,
)

__all__ = [
    "JinaProvider",
    "JinaProviderConfig",
    "JinaProviderSpec",
]
