"""Base class for embedding providers."""

from typing import Any, Generic, TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction

T = TypeVar("T", bound=EmbeddingFunction[Any])


class BaseEmbeddingsProvider(BaseSettings, Generic[T]):
    """Abstract base class for embedding providers.

    This class provides a common interface for dynamically loading and building
    embedding functions from various providers.
    """

    embedding_callable: type[T] = Field(
        ..., description="The embedding function class to use"
    )
