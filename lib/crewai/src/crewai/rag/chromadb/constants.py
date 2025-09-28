"""Constants for ChromaDB configuration."""

import re
from typing import Final

from crewai.utilities.paths import db_storage_path

DEFAULT_TENANT: Final[str] = "default_tenant"
DEFAULT_DATABASE: Final[str] = "default_database"
DEFAULT_STORAGE_PATH: Final[str] = db_storage_path()

MIN_COLLECTION_LENGTH: Final[int] = 3
MAX_COLLECTION_LENGTH: Final[int] = 63
DEFAULT_COLLECTION: Final[str] = "default_collection"

INVALID_CHARS_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9_-]")
IPV4_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
