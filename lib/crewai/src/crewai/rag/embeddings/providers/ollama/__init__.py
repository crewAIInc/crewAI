"""Ollama embedding providers."""

from crewai.rag.embeddings.providers.ollama.ollama_provider import (
    OllamaProvider,
)
from crewai.rag.embeddings.providers.ollama.types import (
    OllamaProviderConfig,
    OllamaProviderSpec,
)

__all__ = [
    "OllamaProvider",
    "OllamaProviderConfig",
    "OllamaProviderSpec",
]
