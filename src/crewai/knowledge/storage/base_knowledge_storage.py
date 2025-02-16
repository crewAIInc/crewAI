from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseKnowledgeStorage(ABC):
    """Abstract base class for knowledge storage implementations."""

    @abstractmethod
    def search(
        self,
        query: List[str],
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Search for documents in the knowledge base."""
        pass

    @abstractmethod
    def save(
        self, documents: List[str], metadata: Dict[str, Any] | List[Dict[str, Any]]
    ) -> None:
        """Save documents to the knowledge base."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the knowledge base."""
        pass
