"""VoyageAI embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.voyageai.types import (
    VoyageAIProviderConfig,
    VoyageAIProviderSpec,
)


__all__ = [
    "VoyageAIProvider",
    "VoyageAIProviderConfig",
    "VoyageAIProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"VoyageAIProvider": f"{__name__}.voyageai_provider"})
