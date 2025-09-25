"""Type definitions for custom embedding providers."""

from typing import Literal, Required, TypedDict

from chromadb.api.types import EmbeddingFunction


class CustomProviderConfig(TypedDict, total=False):
    """Configuration for Custom provider."""

    embedding_callable: Required[type[EmbeddingFunction]]


class CustomProviderSpec(TypedDict):
    """Custom provider specification."""

    provider: Literal["custom"]
    config: CustomProviderConfig
