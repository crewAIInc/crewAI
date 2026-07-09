"""Type definitions specific to turbopuffer implementation."""

from typing import Any, Protocol, TypeAlias

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from turbopuffer import AsyncTurbopuffer, Turbopuffer


TurbopufferClientType: TypeAlias = Turbopuffer | AsyncTurbopuffer


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions that convert text to vectors."""

    def __call__(self, text: str) -> list[float]:
        """Convert text to embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        ...


class AsyncEmbeddingFunction(Protocol):
    """Protocol for async embedding functions that convert text to vectors."""

    async def __call__(self, text: str) -> list[float]:
        """Convert text to embedding vector asynchronously.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        ...


class TurbopufferEmbeddingFunctionWrapper:
    """Pydantic-compatible wrapper for turbopuffer embedding functions."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for turbopuffer EmbeddingFunction."""
        return core_schema.any_schema()


class TurbopufferClientWrapper:
    """Pydantic-compatible wrapper for turbopuffer client instances."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for turbopuffer client."""
        return core_schema.any_schema()
