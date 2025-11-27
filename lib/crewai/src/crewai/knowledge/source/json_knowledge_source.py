import json
from pathlib import Path
from typing import Any

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class JSONKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries JSON file content using embeddings."""

    def load_content(self) -> dict[Path, str]:
        """Load and preprocess JSON file content."""
        content: dict[Path, str] = {}
        for path in self.safe_file_paths:
            path = self.convert_to_path(path)
            with open(path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            content[path] = self._json_to_text(data)
        return content

    def _json_to_text(self, data: Any, level: int = 0) -> str:
        """Recursively convert JSON data to a text representation."""
        text = ""
        indent = "  " * level
        if isinstance(data, dict):
            for key, value in data.items():
                text += f"{indent}{key}: {self._json_to_text(value, level + 1)}\n"
        elif isinstance(data, list):
            for item in data:
                text += f"{indent}- {self._json_to_text(item, level + 1)}\n"
        else:
            text += f"{data!s}"
        return text

    def add(self) -> None:
        """
        Add JSON file content to the knowledge source, chunk it per file,
        attach filepath metadata, and persist via the configured storage.
        """
        for filepath, text in self.content.items():
            chunk_idx = 0
            for chunk in self._chunk_text(text):
                self.chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "filepath": str(filepath),
                            "chunk_index": chunk_idx,
                            "source_type": "json",
                        },
                    }
                )
                chunk_idx += 1
        self._save_documents()  # type: ignore[no-untyped-call]

    def _chunk_text(self, text: str) -> list[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
