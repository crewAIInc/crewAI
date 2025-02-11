from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from urllib.parse import urlparse

from pydantic import Field, field_validator

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger


class ExcelKnowledgeSource(BaseKnowledgeSource):
    """A knowledge source that stores and queries Excel file content using embeddings."""

    # override content to be a dict of file paths to sheet names to csv content

    _logger: Logger = Logger(verbose=True)

    file_path: Optional[Union[Path, List[Path], str, List[str]]] = Field(
        default=None,
        description="[Deprecated] The path to the file. Use file_paths instead.",
    )
    file_paths: Optional[Union[Path, List[Path], str, List[str]]] = Field(
        default_factory=list, description="The path to the file"
    )
    chunks: List[str] = Field(default_factory=list)
    content: Dict[Path, Dict[str, str]] = Field(default_factory=dict)
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

    def model_post_init(self, _) -> None:
        if self.file_path:
            self._logger.log(
                "warning",
                "The 'file_path' attribute is deprecated and will be removed in a future version. Please use 'file_paths' instead.",
                color="yellow",
            )
            self.file_paths = self.file_path
        self.safe_file_paths = self._process_file_paths()
        self.validate_content()
        self.content = self._load_content()

    def _load_content(self) -> Dict[Path, Dict[str, str]]:
        """Load and preprocess Excel file content from multiple sheets.

        Each sheet's content is converted to CSV format and stored.

        Returns:
            Dict[Path, Dict[str, str]]: A mapping of file paths to their respective sheet contents.

        Raises:
            ImportError: If required dependencies are missing.
            FileNotFoundError: If the specified Excel file cannot be opened.
        """
        pd = self._import_dependencies()
        content_dict = {}
        for file_path in self.safe_file_paths:
            file_path = self.convert_to_path(file_path)
            with pd.ExcelFile(file_path) as xl:
                sheet_dict = {
                    str(sheet_name): str(
                        pd.read_excel(xl, sheet_name).to_csv(index=False)
                    )
                    for sheet_name in xl.sheet_names
                }
            content_dict[file_path] = sheet_dict
        return content_dict

    def convert_to_path(self, path: Union[Path, str]) -> Path:
        """Convert a path to a Path object."""
        return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path

    def _import_dependencies(self):
        """Dynamically import dependencies."""
        try:
            import pandas as pd

            return pd
        except ImportError as e:
            missing_package = str(e).split()[-1]
            raise ImportError(
                f"{missing_package} is not installed. Please install it with: pip install {missing_package}"
            )

    def add(self) -> None:
        """
        Add Excel file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        # Convert dictionary values to a single string if content is a dictionary
        # Updated to account for .xlsx workbooks with multiple tabs/sheets
        content_str = ""
        for value in self.content.values():
            if isinstance(value, dict):
                for sheet_value in value.values():
                    content_str += str(sheet_value) + "\n"
            else:
                content_str += str(value) + "\n"

        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
