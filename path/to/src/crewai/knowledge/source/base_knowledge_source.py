from abc import ABC, abstractmethod
from typing import List


class BaseKnowledgeSource(ABC):
    """Abstract base class for different types of knowledge sources."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []

    @abstractmethod
    def load_content(self):
        """Load and preprocess content from the source."""
        pass

    @abstractmethod
    def add(self) -> None:
        """Add content to the knowledge base, chunk it, and compute embeddings."""
        pass

    @abstractmethod
    def query(self, query: str, top_k: int = 3) -> str:
        """Query the knowledge base using semantic search."""
        pass
