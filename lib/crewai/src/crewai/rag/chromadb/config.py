"""ChromaDB configuration model."""

from dataclasses import field
import os
from typing import Annotated, Any, Literal, cast
import warnings

from chromadb.config import Settings
from pydantic import BeforeValidator, ConfigDict, SkipValidation
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


def _coerce_settings(value: Any) -> Settings:
    """Coerce input value to a chromadb.config.Settings instance.

    This validator handles the Pydantic V1/V2 compatibility issue by:
    - Passing through existing Settings objects without validation
    - Converting dict inputs to Settings objects

    Args:
        value: Either a Settings instance or a dict of settings parameters.

    Returns:
        A chromadb.config.Settings instance.

    Raises:
        TypeError: If value is neither a Settings instance nor a dict.
    """
    if isinstance(value, Settings):
        return value
    if isinstance(value, dict):
        return Settings(**value)
    raise TypeError(
        f"settings must be a chromadb.config.Settings instance or a dict, "
        f"got {type(value).__name__}"
    )


# Type alias that skips Pydantic V2 validation for chromadb Settings (Pydantic V1 model)
# and uses a before validator to handle dict-to-Settings conversion
ChromaSettings = Annotated[
    SkipValidation[Settings],
    BeforeValidator(_coerce_settings),
]


def _default_settings() -> Settings:
    """Create default ChromaDB settings.

    Returns:
        Settings with persistent storage and reset enabled.
    """
    return Settings(
        persist_directory=DEFAULT_STORAGE_PATH,
        allow_reset=True,
        is_persistent=True,
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


@pyd_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ChromaDBConfig(BaseRagConfig):
    """Configuration for ChromaDB client.

    The settings field accepts either a chromadb.config.Settings instance
    or a dictionary of settings parameters. This handles the Pydantic V1/V2
    compatibility issue where ChromaDB uses Pydantic V1 for its Settings class.
    """

    provider: Literal["chromadb"] = field(default="chromadb", init=False)
    tenant: str = DEFAULT_TENANT
    database: str = DEFAULT_DATABASE
    settings: ChromaSettings = field(default_factory=_default_settings)
    embedding_function: ChromaEmbeddingFunctionWrapper = field(
        default_factory=_default_embedding_function
    )
