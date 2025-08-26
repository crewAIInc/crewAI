"""Qdrant configuration model."""

import os
from dataclasses import field
from typing import Literal, Any
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig
from crewai.rag.qdrant.types import QdrantClientParams
from crewai.utilities.paths import db_storage_path


def _default_options() -> QdrantClientParams:
    """Create default Qdrant client options.

    Returns:
        Default options with file-based storage.
    """
    return QdrantClientParams(path=os.path.join(db_storage_path(), "qdrant"))


@pyd_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class QdrantConfig(BaseRagConfig):
    """Configuration for Qdrant client."""

    provider: Literal["qdrant"] = field(default="qdrant", init=False)
    options: QdrantClientParams = field(default_factory=_default_options)
    embedding_function: Any | None = None  # Will be EmbeddingFunction when set
