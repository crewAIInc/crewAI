import json
from typing import Any, Dict, List
from pathlib import Path

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class JSONKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries JSON file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess JSON file content."""
        super().load_content()  # Validate the file path
        paths = [self.file_path] if isinstance(self.file_path, Path) else self.file_path

        content: Dict[Path, str] = {}
        for path in paths:
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
            text += f"{str(data)}"
        return text

    def add(self) -> None:
        """
        Add JSON file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        content_str = (
            str(self.content) if isinstance(self.content, dict) else self.content
        )
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self.save_documents(metadata=self.metadata)

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
