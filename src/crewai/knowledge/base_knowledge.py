from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import numpy as np
from .embeddings import Embeddings


class BaseKnowledgeBase(ABC):
    """Abstract base class for knowledge bases"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings_class: Optional[Embeddings] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.chunk_embeddings: Dict[int, np.ndarray] = {}
        self.embeddings_class = embeddings_class or Embeddings()

    @abstractmethod
    def query(self, query: str) -> str:
        """Query the knowledge base and return relevant information"""
        pass

    @abstractmethod
    def add(self, content: Any) -> None:
        """Process and store content in the knowledge base"""
        pass

    def reset(self) -> None:
        """Reset the knowledge base"""
        self.chunks = []
        self.chunk_embeddings = {}

    def _embed_chunks(self, new_chunks: List[str]) -> None:
        """Embed chunks and store them"""
        if not new_chunks:
            return

        # Get embeddings for new chunks
        embeddings = self.embeddings_class.embed_texts(new_chunks)

        # Store embeddings with their corresponding chunks
        start_idx = len(self.chunks)
        for i, embedding in enumerate(embeddings):
            self.chunk_embeddings[start_idx + i] = embedding

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get the chunk of size chunk_size
            end = start + self.chunk_size

            if end >= text_length:
                # If we're at the end, just take the rest
                chunks.append(text[start:].strip())
                break

            # Look for a good breaking point
            # Priority: double newline > single newline > period > space
            break_chars = ["\n\n", "\n", ". ", " "]
            chunk_end = end

            for break_char in break_chars:
                # Look for the break_char in a window around the end point
                window_start = max(start + self.chunk_size - 100, start)
                window_end = min(start + self.chunk_size + 100, text_length)
                window_text = text[window_start:window_end]

                # Find the last occurrence of the break_char in the window
                last_break = window_text.rfind(break_char)
                if last_break != -1:
                    chunk_end = window_start + last_break + len(break_char)
                    break

            # Add the chunk
            chunk = text[start:chunk_end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move the start pointer, accounting for overlap
            start = max(
                start + self.chunk_size - self.chunk_overlap,
                chunk_end - self.chunk_overlap,
            )

        return chunks

    def _find_similar_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Find the most similar chunks to a query using embeddings"""
        if not self.chunks:
            return []

        # Get query embedding
        query_embedding = self.embeddings_class.embed_text(query)

        # Calculate similarities with all chunks
        similarities = []
        for idx, chunk_embedding in self.chunk_embeddings.items():
            similarity = np.dot(query_embedding, chunk_embedding)
            similarities.append((similarity, idx))

        # Sort by similarity and get top_k chunks
        similarities.sort(reverse=True)
        top_chunks = []
        for _, idx in similarities[:top_k]:
            top_chunks.append(self.chunks[idx])

        return top_chunks
