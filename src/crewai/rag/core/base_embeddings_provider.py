"""Base class for embedding providers."""

from typing import Generic, TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction

T = TypeVar("T", bound=EmbeddingFunction)


class BaseEmbeddingsProvider(BaseSettings, Generic[T]):
    """Abstract base class for embedding providers.

    This class provides a common interface for dynamically loading and building
    embedding functions from various providers.
    """

    model_config = SettingsConfigDict(extra="allow", populate_by_name=True)
    embedding_callable: type[T] = Field(
        ..., description="The embedding function class to use"
    )
