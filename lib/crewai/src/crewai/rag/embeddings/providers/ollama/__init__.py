"""Ollama embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.ollama.types import (
    OllamaProviderConfig,
    OllamaProviderSpec,
)


__all__ = [
    "OllamaProvider",
    "OllamaProviderConfig",
    "OllamaProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"OllamaProvider": f"{__name__}.ollama_provider"})
