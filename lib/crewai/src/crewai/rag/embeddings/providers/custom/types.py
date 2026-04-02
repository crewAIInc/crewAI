"""Type definitions for custom embedding providers."""

from typing import Literal

from chromadb.api.types import EmbeddingFunction
from typing_extensions import Required, TypedDict


class CustomProviderConfig(TypedDict, total=False):
    """Configuration for Custom provider."""

    embedding_callable: type[EmbeddingFunction]  # type: ignore[type-arg]


class CustomProviderSpec(TypedDict, total=False):
    """Custom provider specification."""

    provider: Required[Literal["custom"]]
    config: CustomProviderConfig
