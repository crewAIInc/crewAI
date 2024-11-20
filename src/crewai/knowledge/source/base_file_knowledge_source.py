from pathlib import Path
from typing import Union, List

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from typing import Dict, Any
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class BaseFileKnowledgeSource(BaseKnowledgeSource):
    """Base class for knowledge sources that load content from files."""

    file_path: Union[Path, List[Path]] = Field(...)
    content: Dict[Path, str] = Field(init=False, default_factory=dict)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)

    def model_post_init(self, _):
        """Post-initialization method to load content."""
        self.content = self.load_content()

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess file content. Should be overridden by subclasses."""
        paths = [self.file_path] if isinstance(self.file_path, Path) else self.file_path

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
        return {}

    def save_documents(self, metadata: Dict[str, Any]):
        """Save the documents to the storage."""
        chunk_metadatas = [metadata.copy() for _ in self.chunks]
        self.storage.save(self.chunks, chunk_metadatas)
