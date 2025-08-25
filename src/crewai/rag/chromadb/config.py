"""ChromaDB configuration model."""

import os
import warnings
from dataclasses import field
from typing import Literal, cast
from pydantic.dataclasses import dataclass as pyd_dataclass
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from crewai.rag.chromadb.types import ChromaEmbeddingFunctionWrapper
from crewai.utilities.paths import db_storage_path
from crewai.rag.config.base import BaseRagConfig
from crewai.rag.chromadb.constants import DEFAULT_TENANT, DEFAULT_DATABASE

# Suppress Pydantic v1/v2 mixing warning for ChromaDB Settings
warnings.filterwarnings(
    "ignore",
    message=".*Mixing V1 models and V2 models.*",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
)


def _default_settings() -> Settings:
    """Create default ChromaDB settings.

    Returns:
        Settings with persistent storage and reset enabled.
    """
    return Settings(
        persist_directory=os.path.join(db_storage_path(), "chromadb"),
        allow_reset=True,
        is_persistent=True,
    )


def _default_embedding_function() -> ChromaEmbeddingFunctionWrapper:
    """Create default ChromaDB embedding function.

    Returns:
        Default embedding function cast to proper type.
    """
    return cast(ChromaEmbeddingFunctionWrapper, DefaultEmbeddingFunction())


@pyd_dataclass(frozen=True)
class ChromaDBConfig(BaseRagConfig):
    """Configuration for ChromaDB client."""

    provider: Literal["chromadb"] = field(default="chromadb", init=False)
    tenant: str = DEFAULT_TENANT
    database: str = DEFAULT_DATABASE
    settings: Settings = field(default_factory=_default_settings)
    embedding_function: ChromaEmbeddingFunctionWrapper | None = field(
        default_factory=_default_embedding_function
    )
