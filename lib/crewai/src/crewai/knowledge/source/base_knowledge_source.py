from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class BaseKnowledgeSource(BaseModel, ABC):
    """Abstract base class for knowledge sources."""

    chunk_size: int = 4000
    chunk_overlap: int = 200
    chunks: list[str] = Field(default_factory=list)
    chunk_embeddings: list[np.ndarray] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: KnowledgeStorage | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Currently unused
    collection_name: str | None = Field(default=None)

    @abstractmethod
    def validate_content(self) -> Any:
        """Load and preprocess content from the source."""

    @abstractmethod
    def add(self) -> None:
        """Process content, chunk it, compute embeddings, and save them."""

    def get_embeddings(self) -> list[np.ndarray]:
        """Return the list of embeddings for the chunks."""
        return self.chunk_embeddings

    def _chunk_text(self, text: str) -> list[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _save_documents(self) -> None:
        """Save the documents to the storage.

        This method should be called after the chunks and embeddings are generated.

        Raises:
            ValueError: If no storage is configured.
        """
        if self.storage:
            self.storage.save(self.chunks)
        else:
            raise ValueError("No storage found to save documents.")

    @abstractmethod
    async def aadd(self) -> None:
        """Process content, chunk it, compute embeddings, and save them asynchronously."""

    async def _asave_documents(self) -> None:
        """Save the documents to the storage asynchronously.

        This method should be called after the chunks and embeddings are generated.

        Raises:
            ValueError: If no storage is configured.
        """
        if self.storage:
            await self.storage.asave(self.chunks)
        else:
            raise ValueError("No storage found to save documents.")
