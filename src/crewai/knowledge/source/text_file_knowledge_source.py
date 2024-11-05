from crewai.knowledge.embedder.base_embedder import BaseEmbedder
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource


class TextFileKnowledgeSource(BaseKnowledgeSource):
    """A knowledge base that stores and queries plain text content using embeddings"""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        super().__init__(
            chunk_size,
            chunk_overlap,
        )

    def add(self, embedder: BaseEmbedder) -> None:
        """Add text content to the knowledge base, chunk it, and compute embeddings"""
        if not isinstance(self.content, str):
            raise ValueError("StringKnowledgeBase only accepts string content")

        # Create chunks from the text
        new_chunks = self._chunk_text(content)

        # Add chunks to the knowledge base
        self.chunks.extend(new_chunks)

        # Compute and store embeddings for the new chunks
        embedder.embed_chunks(new_chunks)

    def query(self, embedder: BaseEmbedder, query: str, top_k: int = 3) -> str:
        """
        Query the knowledge base using semantic search
        Returns the most relevant chunk based on embedding similarity
        """
        similar_chunks = self._find_similar_chunks(embedder, query, top_k=top_k)
        return similar_chunks[0] if similar_chunks else ""
