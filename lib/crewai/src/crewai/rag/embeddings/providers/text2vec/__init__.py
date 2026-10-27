"""Text2Vec embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.text2vec.types import (
    Text2VecProviderConfig,
    Text2VecProviderSpec,
)


__all__ = [
    "Text2VecProvider",
    "Text2VecProviderConfig",
    "Text2VecProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"Text2VecProvider": f"{__name__}.text2vec_provider"})
