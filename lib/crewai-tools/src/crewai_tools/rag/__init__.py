from typing import Any

from crewai_tools.rag.data_types import DataType
from crewai_tools.rag.embedding_service import EmbeddingService


__all__ = [
    "RAG",
    "DataType",
    "EmbeddingService",
]


def __getattr__(name: str) -> Any:
    if name == "RAG":
        from crewai_tools.rag.core import RAG

        return RAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
