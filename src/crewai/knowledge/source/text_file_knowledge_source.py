from typing import Dict, List
from pathlib import Path

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class TextFileKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries text file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess text file content."""
        super().load_content()
        paths = [self.file_path] if isinstance(self.file_path, Path) else self.file_path
        content = {}
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                content[path] = f.read()  # type: ignore
        return content

    def add(self) -> None:
        """
        Add text file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self.save_documents(metadata=self.metadata)

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
