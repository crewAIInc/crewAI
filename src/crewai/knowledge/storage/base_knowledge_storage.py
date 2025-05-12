from abc import ABC, abstractmethod
from typing import Any


class BaseKnowledgeStorage(ABC):
    """Abstract base class for knowledge storage implementations."""

    @abstractmethod
    def search(
        self,
        query: list[str],
        limit: int = 3,
        filter: dict | None = None,
        score_threshold: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Search for documents in the knowledge base."""

    @abstractmethod
    def save(
        self, documents: list[str], metadata: dict[str, Any] | list[dict[str, Any]],
    ) -> None:
        """Save documents to the knowledge base."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the knowledge base."""
