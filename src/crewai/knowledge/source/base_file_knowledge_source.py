from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.utilities.logger import Logger
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY


class BaseFileKnowledgeSource(BaseKnowledgeSource, ABC):
    """Base class for knowledge sources that load content from files."""

    _logger: Logger = Logger(verbose=True)
    file_path: Union[Path, List[Path], str, List[str]] = Field(
        ..., description="The path to the file"
    )
    content: Dict[Path, str] = Field(init=False, default_factory=dict)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)

    def model_post_init(self, _):
        """Post-initialization method to load content."""
        self.validate_paths()
        self.content = self.load_content()

    @abstractmethod
    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess file content. Should be overridden by subclasses. Assume that the file path is relative to the project root in the knowledge directory."""
        pass

    def validate_paths(self):
        """Validate the paths."""
        if isinstance(self.file_path, str):
            self.file_path = self.convert_to_path(self.file_path)
        elif isinstance(self.file_path, list):
            processed_paths = []
            for path in self.file_path:
                processed_paths.append(self.convert_to_path(path))
            self.file_path = processed_paths

        paths = [self.file_path] if isinstance(self.file_path, Path) else self.file_path
        if not isinstance(paths, list):
            raise ValueError("file_path must be a Path or a list of Paths")

        paths = [Path(path) if isinstance(path, str) else path for path in paths]

        for path in paths:
            if not path.exists():
                self._logger.log(
                    "error",
                    f"File not found: {path}. Try adding sources to the knowledge directory.",
                    color="red",
                )
                raise FileNotFoundError(f"File not found: {path}")
            if not path.is_file():
                self._logger.log(
                    "error",
                    f"Path is not a file: {path}",
                    color="red",
                )

    def save_documents(self, metadata: Dict[str, Any]):
        """Save the documents to the storage."""
        chunk_metadatas = [metadata.copy() for _ in self.chunks]
        self.storage.save(self.chunks, chunk_metadatas)

    def convert_to_path(self, path: Union[Path, str]) -> Path:
        """Convert a path to a Path object."""
        return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path
