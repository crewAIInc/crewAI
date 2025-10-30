import csv
from pathlib import Path

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class CSVKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries CSV file content using embeddings."""

    def load_content(self) -> dict[Path, str]:
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
        Add CSV file content to the knowledge source, chunk it with metadata,
        and save the embeddings.
        """
        for filepath, text in self.content.items():
            text_chunks = self._chunk_text(text)
            for chunk_index, chunk in enumerate(text_chunks):
                self.chunks.append({
                    "content": chunk,
                    "metadata": {
                        "filepath": str(filepath),
                        "chunk_index": chunk_index,
                        "source_type": "csv",
                    }
                })
        self._save_documents()

    def _chunk_text(self, text: str) -> list[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
