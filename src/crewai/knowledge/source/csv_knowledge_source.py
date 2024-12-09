import csv
from pathlib import Path
from typing import Dict, List

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class CSVKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries CSV file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess CSV file content."""
        content_dict = {}
        for file_path in self.safe_file_paths:
            with open(file_path, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                content = ""
                for row in reader:
                    content += " ".join(row) + "\n"
                content_dict[file_path] = content
        return content_dict

    def add(self) -> None:
        """
        Add CSV file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        content_str = (
            str(self.content) if isinstance(self.content, dict) else self.content
        )
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
