"""Base provider protocol for vector database client creation."""

from abc import ABC
from typing import Any, Protocol, runtime_checkable, Union
from pydantic import BaseModel, Field

from crewai.rag.types import EmbeddingFunction
from crewai.rag.embeddings.types import EmbeddingOptions


class BaseProviderOptions(BaseModel, ABC):
    """Base configuration for all provider options."""

    client_type: str = Field(..., description="Type of client to create")
    embedding_config: Union[EmbeddingOptions, EmbeddingFunction, None] = Field(
        default=None,
        description="Embedding configuration - either options for built-in providers or a custom function",
    )
    options: Any = Field(
        default=None, description="Additional provider-specific options"
    )


@runtime_checkable
class BaseProvider(Protocol):
    """Protocol for vector database client providers."""

    def __call__(self, options: BaseProviderOptions) -> Any:
        """Create and return a configured client instance."""
        ...
