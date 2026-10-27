"""OpenCLIP embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.openclip.types import (
    OpenCLIPProviderConfig,
    OpenCLIPProviderSpec,
)


__all__ = [
    "OpenCLIPProvider",
    "OpenCLIPProviderConfig",
    "OpenCLIPProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"OpenCLIPProvider": f"{__name__}.openclip_provider"})
