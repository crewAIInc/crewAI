import os
from abc import ABC, abstractmethod
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger


class BaseFileKnowledgeSource(BaseKnowledgeSource, ABC):
    """
    Base class for knowledge sources that load content from files.
    
    This class provides common functionality for file-based knowledge sources,
    including file path validation, content loading, and change detection.
    It automatically tracks file modification times to detect when files have
    been updated and need to be reloaded.
    
    Attributes:
        file_path: Deprecated. Use file_paths instead.
        file_paths: Path(s) to the file(s) containing knowledge data.
        content: Dictionary mapping file paths to their loaded content.
        storage: Storage backend for the knowledge data.
        safe_file_paths: Validated list of Path objects.
    """

    _logger: Logger = Logger(verbose=True)
    _lock = RLock()  # Thread-safe lock for file operations
    
    file_path: Optional[Union[Path, List[Path], str, List[str]]] = Field(
        default=None,
        description="[Deprecated] The path to the file. Use file_paths instead.",
    )
    file_paths: Optional[Union[Path, List[Path], str, List[str]]] = Field(
        default_factory=list, description="The path to the file"
    )
    content: Dict[Path, str] = Field(init=False, default_factory=dict)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    safe_file_paths: List[Path] = Field(default_factory=list)

    @field_validator("file_path", "file_paths", mode="before")
    def validate_file_path(cls, v, info):
        """Validate that at least one of file_path or file_paths is provided."""
        # Single check if both are None, O(1) instead of nested conditions
        if (
            v is None
            and info.data.get(
                "file_path" if info.field_name == "file_paths" else "file_paths"
            )
            is None
        ):
            raise ValueError("Either file_path or file_paths must be provided")
        return v

    def model_post_init(self, _):
        """Post-initialization method to load content."""
        self.safe_file_paths = self._process_file_paths()
        self.validate_content()
        self._record_file_mtimes()
        self.content = self.load_content()
        
    def _record_file_mtimes(self):
        """
        Record modification times of all files.
        
        This method stores the current modification timestamps of all files
        in the _file_mtimes dictionary. These timestamps are later used to
        detect when files have been modified and need to be reloaded.
        
        Thread-safe: Uses a lock to prevent concurrent modifications.
        """
        with self._lock:
            self._file_mtimes = {}
            for path in self.safe_file_paths:
                try:
                    if path.exists() and path.is_file():
                        if os.access(path, os.R_OK):
                            self._file_mtimes[path] = path.stat().st_mtime
                        else:
                            self._logger.log("warning", f"File {path} is not readable.")
                except PermissionError as e:
                    self._logger.log("error", f"Permission error when recording file timestamp for {path}: {str(e)}")
                except IOError as e:
                    self._logger.log("error", f"IO error when recording file timestamp for {path}: {str(e)}")
                except Exception as e:
                    self._logger.log("error", f"Unexpected error when recording file timestamp for {path}: {str(e)}")

    @abstractmethod
    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess file content. Should be overridden by subclasses. Assume that the file path is relative to the project root in the knowledge directory."""
        pass

    def validate_content(self):
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
        if self.storage:
            self.storage.save(self.chunks)
        else:
            raise ValueError("No storage found to save documents.")

    def convert_to_path(self, path: Union[Path, str]) -> Path:
        """Convert a path to a Path object."""
        return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path

    def _process_file_paths(self) -> List[Path]:
        """Convert file_path to a list of Path objects."""

        if hasattr(self, "file_path") and self.file_path is not None:
            self._logger.log(
                "warning",
                "The 'file_path' attribute is deprecated and will be removed in a future version. Please use 'file_paths' instead.",
                color="yellow",
            )
            self.file_paths = self.file_path

        if self.file_paths is None:
            raise ValueError("Your source must be provided with a file_paths: []")

        # Convert single path to list
        path_list: List[Union[Path, str]] = (
            [self.file_paths]
            if isinstance(self.file_paths, (str, Path))
            else list(self.file_paths)
            if isinstance(self.file_paths, list)
            else []
        )

        if not path_list:
            raise ValueError(
                "file_path/file_paths must be a Path, str, or a list of these types"
            )

        return [self.convert_to_path(path) for path in path_list]
        
    def files_have_changed(self) -> bool:
        """
        Check if any of the files have been modified since they were last loaded.
        
        This method compares the current modification timestamps of files with the
        previously recorded timestamps to detect changes. When a file has been modified,
        it logs the change and returns True to trigger a reload.
        
        Thread-safe: Uses a lock to prevent concurrent modifications.
        
        Returns:
            bool: True if any file has been modified, False otherwise.
        """
        with self._lock:
            for path in self.safe_file_paths:
                try:
                    if not path.exists():
                        self._logger.log("warning", f"File {path} no longer exists.")
                        continue
                        
                    if not path.is_file():
                        self._logger.log("warning", f"Path {path} is not a file.")
                        continue
                        
                    if not os.access(path, os.R_OK):
                        self._logger.log("warning", f"File {path} is not readable.")
                        continue
                        
                    current_mtime = path.stat().st_mtime
                    if path not in self._file_mtimes or current_mtime > self._file_mtimes[path]:
                        self._logger.log("info", f"File {path} has been modified. Reloading data.")
                        return True
                except PermissionError as e:
                    self._logger.log("error", f"Permission error when checking file {path}: {str(e)}")
                except IOError as e:
                    self._logger.log("error", f"IO error when checking file {path}: {str(e)}")
                except Exception as e:
                    self._logger.log("error", f"Unexpected error when checking file {path}: {str(e)}")
                    
            return False
