"""Type definitions for optional imports."""

from typing import Annotated, Literal

SupportedProvider = Annotated[
    Literal["chromadb"],
    "Supported RAG provider types, add providers here as they become available",
]
