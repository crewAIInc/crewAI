"""Constants for ChromaDB configuration."""

import os
from typing import Final

from crewai.utilities.paths import db_storage_path

DEFAULT_TENANT: Final[str] = "default_tenant"
DEFAULT_DATABASE: Final[str] = "default_database"
DEFAULT_STORAGE_PATH: Final[str] = os.path.join(db_storage_path(), "chromadb")
