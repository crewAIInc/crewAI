"""OpenCLIP embedding providers."""

from crewai.rag.embeddings.providers.openclip.openclip_provider import (
    OpenCLIPProvider,
)
from crewai.rag.embeddings.providers.openclip.types import (
    OpenCLIPProviderConfig,
    OpenCLIPProviderSpec,
)

__all__ = [
    "OpenCLIPProvider",
    "OpenCLIPProviderConfig",
    "OpenCLIPProviderSpec",
]
