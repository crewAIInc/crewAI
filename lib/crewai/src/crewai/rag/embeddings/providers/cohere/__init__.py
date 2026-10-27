"""Cohere embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.cohere.types import (
    CohereProviderConfig,
    CohereProviderSpec,
)


__all__ = [
    "CohereProvider",
    "CohereProviderConfig",
    "CohereProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"CohereProvider": f"{__name__}.cohere_provider"})
