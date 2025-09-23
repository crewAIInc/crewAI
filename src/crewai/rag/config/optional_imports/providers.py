"""Provider-specific missing configuration classes."""

from dataclasses import field
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.optional_imports.base import _MissingProvider


@pyd_dataclass(config=ConfigDict(extra="forbid"))
class MissingChromaDBConfig(_MissingProvider):
    """Placeholder for missing ChromaDB configuration."""

    provider: Literal["chromadb"] = field(default="chromadb")


@pyd_dataclass(config=ConfigDict(extra="forbid"))
class MissingQdrantConfig(_MissingProvider):
    """Placeholder for missing Qdrant configuration."""

    provider: Literal["qdrant"] = field(default="qdrant")
