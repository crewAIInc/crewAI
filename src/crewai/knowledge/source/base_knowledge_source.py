from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class BaseKnowledgeSource(BaseModel, ABC):
    """Abstract base class for knowledge sources."""

    chunk_size: int = 4000
    chunk_overlap: int = 200
    chunks: List[str] = Field(default_factory=list)
    chunk_embeddings: List[np.ndarray] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection_name: Optional[str] = Field(default=None)

    @abstractmethod
    def load_content(self) -> Dict[Any, str]:
        """Load and preprocess content from the source."""
        pass

    @abstractmethod
    def add(self) -> None:
        """Process content, chunk it, compute embeddings, and save them."""
        pass

    def get_embeddings(self) -> List[np.ndarray]:
        """Return the list of embeddings for the chunks."""
        return self.chunk_embeddings

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def save_documents(self, metadata: Dict[str, Any]):
        """
        Save the documents to the storage.
        This method should be called after the chunks and embeddings are generated.
        """
        self.storage.save(self.chunks, metadata)
