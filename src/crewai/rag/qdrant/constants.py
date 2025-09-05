"""Constants for Qdrant implementation."""

import os
from typing import Final

from qdrant_client.models import Distance, VectorParams

from crewai.utilities.paths import db_storage_path

DEFAULT_VECTOR_PARAMS: Final = VectorParams(size=384, distance=Distance.COSINE)
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_STORAGE_PATH: Final[str] = os.path.join(db_storage_path(), "qdrant")
