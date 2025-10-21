"""IBM embedding providers."""

from crewai.rag.embeddings.providers.ibm.types import (
    WatsonXProviderConfig,
    WatsonXProviderSpec,
)
from crewai.rag.embeddings.providers.ibm.watsonx import (
    WatsonXProvider,
)


__all__ = [
    "WatsonXProvider",
    "WatsonXProviderConfig",
    "WatsonXProviderSpec",
]
