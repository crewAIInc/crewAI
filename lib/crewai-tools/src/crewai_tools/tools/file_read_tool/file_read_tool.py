"""Tool for reading file contents from disk with line number support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


BINARY_CHECK_SIZE = 8192
MAX_LINE_LENGTH = 500
DEFAULT_LINE_LIMIT = 500


class FileReadToolSchema(BaseModel):
    """Input for FileReadTool."""

    file_path: str = Field(..., description="Mandatory file full path to read the file")
    offset: int | None = Field(
        None,
        description=(
            "Line number to start reading from. Positive values are 1-indexed from "
            "the start. Negative values count from the end (e.g., -10 reads last 10 lines). "
            "If None, reads from the beginning."
        ),
    )
    limit: int | None = Field(
        None,
        description=(
            "Maximum number of lines to read. If None, reads up to the default limit "
            f"({DEFAULT_LINE_LIMIT} lines) for large files, or entire file for small files."
        ),
    )
    include_line_numbers: bool = Field(
        True,
        description="Whether to prefix each line with its line number (format: 'LINE_NUMBER|CONTENT')",
    )
    start_line: int | None = Field(
        None,
        description="[DEPRECATED: Use 'offset' instead] Line number to start reading from (1-indexed).",
    )
    line_count: int | None = Field(
        None,
        description="[DEPRECATED: Use 'limit' instead] Number of lines to read.",
    )


class FileReadTool(BaseTool):
    """A tool for reading file contents with line number support.

    This tool provides Claude Code-like file reading capabilities:
    - Line number prefixes for easy reference
    - Offset/limit support for reading specific portions of large files
    - Negative offset support for reading from end of file
    - Binary file detection
    - File metadata (total lines) in response header

    The tool supports two ways of specifying the file path:
    1. At construction time via the file_path parameter
    2. At runtime via the file_path parameter in the tool's input

    Args:
        file_path (Optional[str]): Path to the file to be read. If provided,
            this becomes the default file path for the tool.
        **kwargs: Additional keyword arguments passed to BaseTool.

    Example:
        >>> tool = FileReadTool()
        >>> content = tool.run(file_path="/path/to/file.txt")  # Reads entire file
        >>> content = tool.run(
        ...     file_path="/path/to/file.txt", offset=100, limit=50
        ... )  # Lines 100-149
        >>> content = tool.run(
        ...     file_path="/path/to/file.txt", offset=-20
        ... )  # Last 20 lines
    """

    name: str = "read_file"
    description: str = (
        "Read content from a file on disk. Returns file content with line numbers "
        "prefixed (format: 'LINE_NUMBER|CONTENT'). Use 'offset' to start from a "
        "specific line (negative values read from end), and 'limit' to control "
        "how many lines to read. For large files, reads are automatically limited."
    )
    args_schema: type[BaseModel] = FileReadToolSchema
    file_path: str | None = None

    def __init__(self, file_path: str | None = None, **kwargs: Any) -> None:
        """Initialize the FileReadTool.

        Args:
            file_path (Optional[str]): Path to the file to be read. If provided,
                this becomes the default file path for the tool.
            **kwargs: Additional keyword arguments passed to BaseTool.
        """
        if file_path is not None:
            kwargs["description"] = (
                f"Read content from a file. The default file is {file_path}, but you "
                "can provide a different 'file_path' parameter. Use 'offset' to start "
                "from a specific line and 'limit' to control the number of lines read."
            )

        super().__init__(**kwargs)
        self.file_path = file_path

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if a file is binary by looking for null bytes.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file appears to be binary.
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(BINARY_CHECK_SIZE)
                return b"\x00" in chunk
        except (OSError, PermissionError):
            return True

    def _count_lines(self, file_path: Path) -> int:
        """Count total lines in a file efficiently.

        Args:
            file_path: Path to the file.

        Returns:
            Total number of lines in the file.
        """
        try:
            with open(file_path, "rb") as f:
                return sum(1 for _ in f)
        except (OSError, PermissionError):
            return 0

    def _run(
        self,
        file_path: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        include_line_numbers: bool = True,
        start_line: int | None = None,
        line_count: int | None = None,
    ) -> str:
        """Read file contents with optional line range.

        Args:
            file_path: Path to the file to read.
            offset: Line to start from (1-indexed, negative counts from end).
            limit: Maximum lines to read.
            include_line_numbers: Whether to prefix lines with numbers.
            start_line: Legacy parameter (maps to offset).
            line_count: Legacy parameter (maps to limit).

        Returns:
            File content with metadata header.
        """
        if start_line is not None and offset is None:
            offset = start_line
        if line_count is not None and limit is None:
            limit = line_count

        file_path = file_path or self.file_path

        if file_path is None:
            return "Error: No file path provided. Please provide a file path either in the constructor or as an argument."

        path = Path(file_path)

        if not path.exists():
            return f"Error: File not found at path: {file_path}"

        if path.is_dir():
            return f"Error: Path is a directory, not a file: {file_path}"

        if self._is_binary_file(path):
            file_size = path.stat().st_size
            return (
                f"Error: '{file_path}' appears to be a binary file ({file_size} bytes). "
                "Binary files cannot be read as text. Use a specialized tool for binary content."
            )

        try:
            total_lines = self._count_lines(path)

            if total_lines == 0:
                return f"File: {file_path}\nTotal lines: 0\n\n(Empty file)"

            if offset is None:
                start_idx = 0
            elif offset < 0:
                start_idx = max(0, total_lines + offset)
            else:
                start_idx = max(0, offset - 1)

            if limit is None:
                if total_lines > DEFAULT_LINE_LIMIT and offset is None:
                    effective_limit = DEFAULT_LINE_LIMIT
                else:
                    effective_limit = total_lines - start_idx
            else:
                effective_limit = limit

            end_idx = min(start_idx + effective_limit, total_lines)

            with open(path, encoding="utf-8", errors="replace") as f:
                lines: list[str] = []
                for i, line in enumerate(f):
                    if i < start_idx:
                        continue
                    if i >= end_idx:
                        break

                    line_content = line.rstrip("\n\r")

                    if len(line_content) > MAX_LINE_LENGTH:
                        line_content = line_content[:MAX_LINE_LENGTH] + "..."

                    if include_line_numbers:
                        line_num = i + 1  # 1-indexed
                        lines.append(f"{line_num:6}|{line_content}")
                    else:
                        lines.append(line_content)

            header_parts = [f"File: {file_path}", f"Total lines: {total_lines}"]

            if start_idx > 0 or end_idx < total_lines:
                header_parts.append(f"Showing lines: {start_idx + 1}-{end_idx}")

            if end_idx < total_lines and limit is None and offset is None:
                header_parts.append(
                    "(File truncated. Use 'offset' and 'limit' to read more.)"
                )

            header = "\n".join(header_parts)
            content = "\n".join(lines)

            return f"{header}\n\n{content}"

        except PermissionError:
            return f"Error: Permission denied when trying to read file: {file_path}"
        except UnicodeDecodeError as e:
            return f"Error: Failed to decode file {file_path} as text: {e!s}"
        except Exception as e:
            return f"Error: Failed to read file {file_path}. {e!s}"
