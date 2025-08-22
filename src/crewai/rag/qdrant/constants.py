"""Constants for Qdrant implementation."""

from typing import Final

from qdrant_client.models import Distance, VectorParams

DEFAULT_VECTOR_PARAMS: Final = VectorParams(size=1536, distance=Distance.COSINE)
