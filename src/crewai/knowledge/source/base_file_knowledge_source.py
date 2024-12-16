from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger


class BaseFileKnowledgeSource(BaseKnowledgeSource, ABC):
    """Base class for knowledge sources that load content from files."""

    _logger: Logger = Logger(verbose=True)
    file_paths: Union[Path, List[Path], str, List[str]] = Field(
        ..., description="The path to the file"
    )
    content: Dict[Path, str] = Field(init=False, default_factory=dict)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)
    safe_file_paths: List[Path] = Field(default_factory=list)

    def model_post_init(self, _):
        """Post-initialization method to load content."""
        self.safe_file_paths = self._process_file_paths()
        self.validate_paths()
        self.content = self.load_content()

    @abstractmethod
    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess file content. Should be overridden by subclasses. Assume that the file path is relative to the project root in the knowledge directory."""
        pass

    def validate_paths(self):
        """Validate the paths."""
        for path in self.safe_file_paths:
            if not path.exists():
                self._logger.log(
                    "error",
                    f"File not found: {path}. Try adding sources to the knowledge directory. If it's inside the knowledge directory, use the relative path.",
                    color="red",
                )
                raise FileNotFoundError(f"File not found: {path}")
            if not path.is_file():
                self._logger.log(
                    "error",
                    f"Path is not a file: {path}",
                    color="red",
                )

    def _save_documents(self):
        """Save the documents to the storage."""
        self.storage.save(self.chunks)

    def convert_to_path(self, path: Union[Path, str]) -> Path:
        """Convert a path to a Path object."""
        return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path

    def _process_file_paths(self) -> List[Path]:
        """Convert file_path to a list of Path objects."""
        paths = (
            [self.file_paths]
            if isinstance(self.file_paths, (str, Path))
            else self.file_paths
        )

        if not isinstance(paths, list):
            raise ValueError("file_path must be a Path, str, or a list of these types")

        return [self.convert_to_path(path) for path in paths]
