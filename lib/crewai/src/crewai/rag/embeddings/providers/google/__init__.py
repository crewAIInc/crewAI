"""Google embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.google.types import (
    GenerativeAiProviderConfig,
    GenerativeAiProviderSpec,
    VertexAIProviderConfig,
    VertexAIProviderSpec,
)


__all__ = [
    "GenerativeAiProvider",
    "GenerativeAiProviderConfig",
    "GenerativeAiProviderSpec",
    "GoogleGenAIVertexEmbeddingFunction",
    "VertexAIProvider",
    "VertexAIProviderConfig",
    "VertexAIProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(
        name,
        {
            "GenerativeAiProvider": f"{__name__}.generative_ai",
            "GoogleGenAIVertexEmbeddingFunction": f"{__name__}.genai_vertex_embedding",
            "VertexAIProvider": f"{__name__}.vertex",
        },
    )
