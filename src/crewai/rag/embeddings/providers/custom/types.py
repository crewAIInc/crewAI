"""Type definitions for custom embedding providers."""

from typing import Literal

from chromadb.api.types import EmbeddingFunction
from typing_extensions import Required, TypedDict


class CustomProviderConfig(TypedDict, total=False):
    """Configuration for Custom provider."""

    embedding_callable: type[EmbeddingFunction]


class CustomProviderSpec(TypedDict, total=False):
    """Custom provider specification."""

    provider: Required[Literal["custom"]]
    config: CustomProviderConfig
