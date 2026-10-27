"""HuggingFace embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.huggingface.types import (
    HuggingFaceProviderConfig,
    HuggingFaceProviderSpec,
)


__all__ = [
    "HuggingFaceProvider",
    "HuggingFaceProviderConfig",
    "HuggingFaceProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(
        name, {"HuggingFaceProvider": f"{__name__}.huggingface_provider"}
    )
