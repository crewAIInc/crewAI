from pathlib import Path
from typing import Dict, List

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class TextFileKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries text file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess text file content."""
        content = {}
        for path in self.safe_file_paths:
            path = self.convert_to_path(path)
            with open(path, "r", encoding="utf-8") as f:
                content[path] = f.read()
        return content

    def add(self) -> None:
        """
        Add text file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
