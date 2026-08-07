from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


# ``KnowledgeStorage`` is re-exported for backwards compatibility; the ``storage``
# field below is typed to the base interface so any backend plugs in.
__all__ = ["BaseKnowledgeSource", "KnowledgeStorage"]


class BaseKnowledgeSource(BaseModel, ABC):
    """Abstract base class for knowledge sources."""

    chunk_size: int = 4000
    chunk_overlap: int = 200
    chunks: list[str] = Field(default_factory=list)
    chunk_embeddings: list[np.ndarray[Any, np.dtype[Any]]] = Field(
        default_factory=list, exclude=True
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: BaseKnowledgeStorage | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Currently unused
    collection_name: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> Self:
        """Reject a chunk_overlap that would break _chunk_text's slicing step.

        step = chunk_size - chunk_overlap: zero raises ValueError("range() arg 3
        must not be zero"); negative silently makes range() empty, dropping all
        content with no error at all.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        return self

    @abstractmethod
    def validate_content(self) -> Any:
        """Load and preprocess content from the source."""

    @abstractmethod
    def add(self) -> None:
        """Process content, chunk it, compute embeddings, and save them."""

    def get_embeddings(self) -> list[np.ndarray[Any, np.dtype[Any]]]:
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
        if self.storage is not None:
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
        if self.storage is not None:
            await self.storage.asave(self.chunks)
        else:
            raise ValueError("No storage found to save documents.")
