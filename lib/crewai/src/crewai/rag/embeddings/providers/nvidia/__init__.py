"""NVIDIA embeddings provider."""

from crewai.rag.embeddings.providers.nvidia.embedding_callable import (
    NvidiaEmbeddingFunction,
)
from crewai.rag.embeddings.providers.nvidia.nvidia_provider import NvidiaProvider
from crewai.rag.embeddings.providers.nvidia.types import (
    NvidiaEmbeddingModels,
    NvidiaProviderConfig,
    NvidiaProviderSpec,
)

__all__ = [
    "NvidiaProvider",
    "NvidiaEmbeddingFunction",
    "NvidiaEmbeddingModels",
    "NvidiaProviderConfig",
    "NvidiaProviderSpec",
]
