from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource


class StringKnowledgeSource(BaseKnowledgeSource):
    """A knowledge source that stores and queries plain text content using embeddings."""

    content: str = Field(...)
    collection_name: str | None = Field(default=None)

    def model_post_init(self, _):
        """Post-initialization method to validate content."""
        self.validate_content()

    def validate_content(self):
        """Validate string content."""
        if not isinstance(self.content, str):
            raise ValueError("StringKnowledgeSource only accepts string content")

    def add(self) -> None:
        """Add string content to the knowledge source, chunk it with metadata, and save them."""
        text_chunks = self._chunk_text(self.content)
        for chunk_index, chunk in enumerate(text_chunks):
            self.chunks.append({
                "content": chunk,
                "metadata": {
                    "chunk_index": chunk_index,
                    "source_type": "string",
                }
            })
        self._save_documents()

    def _chunk_text(self, text: str) -> list[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
