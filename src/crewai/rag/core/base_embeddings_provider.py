"""Base class for embedding providers."""

from typing import Generic, TypeVar

from chromadb.api.types import EmbeddingFunction
from pydantic import Field
from pydantic_settings import BaseSettings

T = TypeVar("T", bound=EmbeddingFunction)


class BaseEmbeddingsProvider(BaseSettings, Generic[T]):
    """Abstract base class for embedding providers.

    This class provides a common interface for dynamically loading and building
    embedding functions from various providers.
    """

    embedding_callable: type[T] = Field(
        ..., description="The embedding function class to use"
    )


def build_embedder_from_provider(provider: BaseEmbeddingsProvider[T]) -> T:
    """Build an embedding function instance from a provider.

    Args:
        provider: The embedding provider configuration.

    Returns:
        An instance of the specified embedding function type.
    """
    return provider.embedding_callable(
        **provider.model_dump(exclude={"embedding_callable"})
    )
