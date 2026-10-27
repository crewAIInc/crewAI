"""SentenceTransformer embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.sentence_transformer.types import (
    SentenceTransformerProviderConfig,
    SentenceTransformerProviderSpec,
)


__all__ = [
    "SentenceTransformerProvider",
    "SentenceTransformerProviderConfig",
    "SentenceTransformerProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(
        name,
        {"SentenceTransformerProvider": f"{__name__}.sentence_transformer_provider"},
    )
