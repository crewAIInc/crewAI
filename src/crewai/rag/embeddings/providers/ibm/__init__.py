"""IBM embedding providers."""

from crewai.rag.embeddings.providers.ibm.types import (
    WatsonProviderConfig,
    WatsonProviderSpec,
)
from crewai.rag.embeddings.providers.ibm.watson import (
    WatsonProvider,
)

__all__ = [
    "WatsonProvider",
    "WatsonProviderConfig",
    "WatsonProviderSpec",
]
