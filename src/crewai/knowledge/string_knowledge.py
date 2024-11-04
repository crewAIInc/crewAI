from typing import Optional
from crewai.knowledge.base_knowledge import BaseKnowledgeBase
from crewai.knowledge.embeddings import Embeddings


class StringKnowledgeBase(BaseKnowledgeBase):
    """A knowledge base that stores and queries plain text content using embeddings"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings_class: Optional[Embeddings] = None,
        content: Optional[str] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, embeddings_class)
        if content:
            self.add(content)

    def add(self, content: str) -> None:
        """Add text content to the knowledge base, chunk it, and compute embeddings"""
        if not isinstance(content, str):
            raise ValueError("StringKnowledgeBase only accepts string content")

        # Create chunks from the text
        new_chunks = self._chunk_text(content)

        # Add chunks to the knowledge base
        self.chunks.extend(new_chunks)

        # Compute and store embeddings for the new chunks
        self._embed_chunks(new_chunks)

    def query(self, query: str, top_k: int = 3) -> str:
        """
        Query the knowledge base using semantic search
        Returns the most relevant chunk based on embedding similarity
        """
        similar_chunks = self._find_similar_chunks(query, top_k=top_k)
        return similar_chunks[0] if similar_chunks else ""
