"""Base configuration class for RAG providers."""

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.optional_imports.types import SupportedProvider


@pyd_dataclass(frozen=True)
class BaseRagConfig:
    """Base class for RAG configuration with Pydantic serialization support."""

    provider: SupportedProvider = field(init=False)
    embedding_function: Any | None = field(default=None)
    limit: int = field(default=5)
    score_threshold: float = field(default=0.6)
    batch_size: int = field(default=100)
