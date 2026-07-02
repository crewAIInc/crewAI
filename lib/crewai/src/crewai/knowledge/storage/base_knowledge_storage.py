from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict


if TYPE_CHECKING:
    from crewai.rag.types import SearchResult


class BaseKnowledgeStorage(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
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
    def save(
        self,
        documents: list[str],
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        """Save documents to the knowledge base.

        Args:
            documents: List of document strings to save.
            metadata: Optional metadata to attach to each stored document.
                A single dict is applied to every document; a list of dicts
                must match the number of documents.
        """

    @abstractmethod
    async def asave(
        self,
        documents: list[str],
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        """Save documents to the knowledge base asynchronously.

        Args:
            documents: List of document strings to save.
            metadata: Optional metadata to attach to each stored document.
                A single dict is applied to every document; a list of dicts
                must match the number of documents.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the knowledge base."""

    @abstractmethod
    async def areset(self) -> None:
        """Reset the knowledge base asynchronously."""
