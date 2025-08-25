"""Base configuration class for RAG providers."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from crewai.rag.config.schema_utils import create_dataclass_schema
from crewai.rag.types import EmbeddingFunction


@dataclass
class BaseRagConfig:
    """Base class for RAG configuration with Pydantic serialization support."""

    embedding_function: EmbeddingFunction | None = field(default=None)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for dataclass serialization.

        Args:
            _source_type: Source type, unused.
            handler: Pydantic's schema handler.

        Returns:
            CoreSchema for validation and serialization.
        """
        return create_dataclass_schema(cls, handler)
