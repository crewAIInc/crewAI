"""AWS embedding providers."""

from crewai.rag.embeddings.providers.aws.bedrock import BedrockProvider
from crewai.rag.embeddings.providers.aws.types import (
    BedrockProviderConfig,
    BedrockProviderSpec,
)


__all__ = [
    "BedrockProvider",
    "BedrockProviderConfig",
    "BedrockProviderSpec",
]
