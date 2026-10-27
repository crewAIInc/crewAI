"""Microsoft embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.microsoft.types import (
    AzureProviderConfig,
    AzureProviderSpec,
)


__all__ = [
    "AzureProvider",
    "AzureProviderConfig",
    "AzureProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"AzureProvider": f"{__name__}.azure"})
