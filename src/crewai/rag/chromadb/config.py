"""ChromaDB configuration model."""

import os
import warnings
from chromadb.config import Settings
from pydantic import BaseModel, Field

from crewai.utilities.paths import db_storage_path

warnings.filterwarnings(
    "ignore",
    message="Mixing V1 models and V2 models",
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


class ChromaDBConfig(BaseModel):
    """Configuration for ChromaDB client."""

    tenant: str = Field(default="default_tenant", description="ChromaDB tenant")
    database: str = Field(default="default_database", description="ChromaDB database")
    settings: Settings = Field(
        default_factory=_default_settings, description="ChromaDB Settings object"
    )
