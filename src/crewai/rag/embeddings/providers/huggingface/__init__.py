"""HuggingFace embedding providers."""

from crewai.rag.embeddings.providers.huggingface.huggingface_provider import (
    HuggingFaceProvider,
)
from crewai.rag.embeddings.providers.huggingface.types import (
    HuggingFaceProviderConfig,
    HuggingFaceProviderSpec,
)

__all__ = [
    "HuggingFaceProvider",
    "HuggingFaceProviderConfig",
    "HuggingFaceProviderSpec",
]
