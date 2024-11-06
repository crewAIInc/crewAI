from abc import ABC, abstractmethod
from typing import List

import numpy as np

from crewai.knowledge.embedder.base_embedder import BaseEmbedder


class BaseKnowledgeSource(ABC):
    """Abstract base class for knowledge sources."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.chunk_embeddings: List[np.ndarray] = []

    @abstractmethod
    def load_content(self):
        """Load and preprocess content from the source."""
        pass

    @abstractmethod
    def add(self, embedder: BaseEmbedder) -> None:
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
