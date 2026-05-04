"""Helper utilities for knowledge sources."""

from typing import Any, ClassVar

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


class SourceHelper:
    """Helper class for creating and managing knowledge sources."""

    SUPPORTED_FILE_TYPES: ClassVar[list[str]] = [
        ".csv",
        ".pdf",
        ".json",
        ".txt",
        ".xlsx",
        ".xls",
    ]

    _FILE_TYPE_MAP: ClassVar[dict[str, type[BaseKnowledgeSource]]] = {
        ".csv": CSVKnowledgeSource,
        ".pdf": PDFKnowledgeSource,
        ".json": JSONKnowledgeSource,
        ".txt": TextFileKnowledgeSource,
        ".xlsx": ExcelKnowledgeSource,
        ".xls": ExcelKnowledgeSource,
    }

    @classmethod
    def is_supported_file(cls, file_path: str) -> bool:
        """Check if a file type is supported.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file type is supported.
        """
        return file_path.lower().endswith(tuple(cls.SUPPORTED_FILE_TYPES))

    @classmethod
    def get_source(
        cls, file_path: str, metadata: dict[str, Any] | None = None
    ) -> BaseKnowledgeSource:
        """Create appropriate KnowledgeSource based on file extension.

        Args:
            file_path: Path to the file.
            metadata: Optional metadata to attach to the source.

        Returns:
            The appropriate KnowledgeSource instance.

        Raises:
            ValueError: If the file type is not supported.
        """
        if not cls.is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")

        lower_path = file_path.lower()
        for ext, source_cls in cls._FILE_TYPE_MAP.items():
            if lower_path.endswith(ext):
                return source_cls(file_path=[file_path], metadata=metadata)

        raise ValueError(f"Unsupported file type: {file_path}")
