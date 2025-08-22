"""ChromaDB configuration model."""

import os
from dataclasses import dataclass, field
from typing import Literal
from chromadb.config import Settings


from crewai.utilities.paths import db_storage_path
from crewai.rag.config.base import BaseRagConfig


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


@dataclass
class ChromaDBConfig(BaseRagConfig):
    """Configuration for ChromaDB client."""

    provider: Literal["chromadb"] = "chromadb"
    tenant: str = "default_tenant"
    database: str = "default_database"
    settings: Settings = field(default_factory=_default_settings)
