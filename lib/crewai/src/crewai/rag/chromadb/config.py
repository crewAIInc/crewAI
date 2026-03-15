"""ChromaDB configuration model."""

from dataclasses import field
import os
from typing import Literal, cast
import warnings

from chromadb.config import Settings
from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.chromadb.constants import (
    DEFAULT_DATABASE,
    DEFAULT_STORAGE_PATH,
    DEFAULT_TENANT,
)
from crewai.rag.chromadb.types import ChromaEmbeddingFunctionWrapper
from crewai.rag.config.base import BaseRagConfig


warnings.filterwarnings(
    "ignore",
    message=".*Mixing V1 models and V2 models.*",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
)

warnings.filterwarnings(
    "ignore",
    message=r".*'model_fields'.*is deprecated.*",
    module=r"^chromadb(\.|$)",
)


def _default_settings() -> Settings:
    """Create default ChromaDB settings.

    Returns:
        Settings with persistent storage and reset enabled.
    """
    return Settings(
        persist_directory=DEFAULT_STORAGE_PATH,
        allow_reset=True,
        is_persistent=True,
        anonymized_telemetry=False,
    )


def _default_embedding_function() -> ChromaEmbeddingFunctionWrapper:
    """Create default ChromaDB embedding function.

    Returns:
        Default embedding function using all-MiniLM-L6-v2 via ONNX.
    """
    from chromadb.utils.embedding_functions.openai_embedding_function import (
        OpenAIEmbeddingFunction,
    )

    return cast(
        ChromaEmbeddingFunctionWrapper,
        OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
            api_key_env_var="OPENAI_API_KEY",
        ),
    )


@pyd_dataclass(frozen=True)
class ChromaDBConfig(BaseRagConfig):
    """Configuration for ChromaDB client."""

    provider: Literal["chromadb"] = field(default="chromadb", init=False)
    tenant: str = DEFAULT_TENANT
    database: str = DEFAULT_DATABASE
    settings: Settings = field(default_factory=_default_settings)
    embedding_function: ChromaEmbeddingFunctionWrapper = field(
        default_factory=_default_embedding_function
    )
