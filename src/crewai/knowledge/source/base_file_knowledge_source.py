from pathlib import Path

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from typing import Dict, Any


class BaseFileKnowledgeSource(BaseKnowledgeSource):
    """Base class for knowledge sources that load content from files."""

    file_path: Path = Field(...)
    content: str = Field(init=False, default="")

    def model_post_init(self, context):
        """Post-initialization method to load content."""
        self.content = self.load_content()

    def load_content(self) -> str:
        """Load and preprocess file content. Should be overridden by subclasses."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")
        return ""

    def _save_documents(self, metadata: Dict[str, Any]):
        """Save the documents to the storage."""
        self.storage.save(self.chunks, metadata)
