from typing import Any

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource


class StringKnowledgeSource(BaseKnowledgeSource):
    """A knowledge source that stores and queries plain text content using embeddings."""

    content: str = Field(...)
    collection_name: str | None = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization method to validate content."""
        self.validate_content()

    def validate_content(self) -> None:
        """Validate string content."""
        if not isinstance(self.content, str):
            raise ValueError("StringKnowledgeSource only accepts string content")

    def add(self) -> None:
        """
        Add string content to the knowledge source, chunk it,
        attach metadata, and persist via the configured storage.
        """
        chunk_idx = 0
        for chunk in self._chunk_text(self.content):
            metadata: dict[str, Any] = {
                "source_type": "string",
                "chunk_index": chunk_idx,
            }
            self.chunks.append(
                {
                    "content": chunk,
                    "metadata": metadata,
                }
            )
            chunk_idx += 1
        self._save_documents()

    def _chunk_text(self, text: str) -> list[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
