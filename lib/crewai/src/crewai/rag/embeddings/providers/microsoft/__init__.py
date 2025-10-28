"""Microsoft embedding providers."""

from crewai.rag.embeddings.providers.microsoft.azure import (
    AzureProvider,
)
from crewai.rag.embeddings.providers.microsoft.types import (
    AzureProviderConfig,
    AzureProviderSpec,
)


__all__ = [
    "AzureProvider",
    "AzureProviderConfig",
    "AzureProviderSpec",
]
