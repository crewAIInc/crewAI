from abc import ABC, abstractmethod
from typing import Any

from crewai.rag.types import SearchResult


class BaseKnowledgeStorage(ABC):
    """Abstract base class for knowledge storage implementations."""

    @abstractmethod
    def search(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        """Search for documents in the knowledge base."""

    @abstractmethod
    def save(self, documents: list[str]) -> None:
        """Save documents to the knowledge base."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the knowledge base."""
