"""OpenAI embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.openai.types import (
    OpenAIProviderConfig,
    OpenAIProviderSpec,
)


__all__ = [
    "OpenAIProvider",
    "OpenAIProviderConfig",
    "OpenAIProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"OpenAIProvider": f"{__name__}.openai_provider"})
