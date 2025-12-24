from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class BaseKnowledgeSource(BaseModel, ABC):
    """Abstracte basis klasse voor kennisbronnen."""

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
        """Laad en verwerk content uit de bron."""

    @abstractmethod
    def add(self) -> None:
        """Verwerk content, chunk het, bereken embeddings, en sla ze op."""

    def get_embeddings(self) -> list[np.ndarray]:
        """Retourneer de lijst van embeddings voor de chunks."""
        return self.chunk_embeddings

    def _chunk_text(self, text: str) -> list[str]:
        """Hulpmethode om tekst op te splitsen in chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _save_documents(self) -> None:
        """Sla de documenten op in de opslag.

        Deze methode moet worden aangeroepen nadat de chunks en embeddings zijn gegenereerd.

        Gooit:
            ValueError: Als geen opslag is geconfigureerd.
        """
        if self.storage:
            self.storage.save(self.chunks)
        else:
            raise ValueError("Geen opslag gevonden om documenten op te slaan.")

    @abstractmethod
    async def aadd(self) -> None:
        """Verwerk content, chunk het, bereken embeddings, en sla ze asynchroon op."""

    async def _asave_documents(self) -> None:
        """Sla de documenten asynchroon op in de opslag.

        Deze methode moet worden aangeroepen nadat de chunks en embeddings zijn gegenereerd.

        Gooit:
            ValueError: Als geen opslag is geconfigureerd.
        """
        if self.storage:
            await self.storage.asave(self.chunks)
        else:
            raise ValueError("Geen opslag gevonden om documenten op te slaan.")
