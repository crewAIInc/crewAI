"""Custom embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.custom.types import (
    CustomProviderConfig,
    CustomProviderSpec,
)


__all__ = [
    "CustomProvider",
    "CustomProviderConfig",
    "CustomProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"CustomProvider": f"{__name__}.custom_provider"})
