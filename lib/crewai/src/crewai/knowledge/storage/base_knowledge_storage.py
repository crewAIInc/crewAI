from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
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
    async def asearch(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        """Search for documents in the knowledge base asynchronously."""

    @abstractmethod
    def save(self, documents: list[str]) -> None:
        """Save documents to the knowledge base."""

    @abstractmethod
    async def asave(self, documents: list[str]) -> None:
        """Save documents to the knowledge base asynchronously."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the knowledge base."""

    @abstractmethod
    async def areset(self) -> None:
        """Reset the knowledge base asynchronously."""
