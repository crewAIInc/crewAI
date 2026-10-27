"""AWS embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.aws.types import (
    BedrockProviderConfig,
    BedrockProviderSpec,
)


__all__ = [
    "BedrockProvider",
    "BedrockProviderConfig",
    "BedrockProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"BedrockProvider": f"{__name__}.bedrock"})
