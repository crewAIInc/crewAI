"""Type definitions for optional imports."""

from typing import Annotated, Literal


SupportedProvider = Annotated[
    Literal["chromadb", "qdrant", "turbopuffer"],
    "Supported RAG provider types, add providers here as they become available",
]
