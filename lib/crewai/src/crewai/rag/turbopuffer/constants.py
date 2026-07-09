"""Constants for turbopuffer implementation."""

import re
from typing import Final, Literal


DistanceMetric = Literal["cosine_distance"]

DEFAULT_DISTANCE_METRIC: Final[DistanceMetric] = "cosine_distance"
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
NAMESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9\-_.]{1,128}$")
CONTENT_KEY: Final[str] = "content"
