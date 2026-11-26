from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
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

    @abstractmethod
    def validate_content(self) -> Any:
        """Load and preprocess content from the source."""

    @abstractmethod
    def add(self) -> None:
        """Process content, chunk it, compute embeddings, and save them."""

    def get_embeddings(self) -> list[np.ndarray[Any, np.dtype[Any]]]:
        """Return the list of embeddings for the chunks."""
        return self.chunk_embeddings

    def clone_for_copy(self, *, storage: BaseKnowledgeStorage | None) -> Self:
        """Return a copy that does not share this source's chunk buffers.

        ``add()`` extends ``chunks`` and ``chunk_embeddings`` in place, so the
        copy starts those lists empty.
        """
        copied = self.model_copy()
        copied.chunks = []
        copied.chunk_embeddings = []
        copied.metadata = dict(self.metadata)
        copied.storage = storage
        return copied

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks of ``chunk_size`` that overlap by ``chunk_overlap``.

        Stops once a chunk reaches the end of the text, so the last chunk is
        never just the tail of the one before it.
        """
        if not text:
            return []
        return [
            text[i : i + self.chunk_size]
            for i in range(
                0,
                max(len(text) - self.chunk_overlap, 1),
                self.chunk_size - self.chunk_overlap,
            )
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
