"""OpenRouter embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.openrouter.types import (
    OpenRouterProviderConfig,
    OpenRouterProviderSpec,
)


__all__ = [
    "OpenRouterProvider",
    "OpenRouterProviderConfig",
    "OpenRouterProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(
        name, {"OpenRouterProvider": f"{__name__}.openrouter_provider"}
    )
