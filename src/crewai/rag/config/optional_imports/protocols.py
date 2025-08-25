"""Protocol definitions for RAG factory modules."""

from typing import Protocol

from crewai.rag.config.types import RagConfigType
from crewai.rag.core.base_client import BaseClient


class ChromaFactoryModule(Protocol):
    """Protocol for ChromaDB factory module."""

    def create_client(self, config: RagConfigType) -> BaseClient:
        """Creates a ChromaDB client from configuration."""
        ...
