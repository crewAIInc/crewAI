"""Tool for reading file contents."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from crewai.experimental.environment_tools.base_environment_tool import (
    BaseEnvironmentTool,
)


class FileReadInput(BaseModel):
    """Input schema for reading files."""

    path: str = Field(..., description="Path to the file to read")
    start_line: int | None = Field(
        default=None,
        description="Line to start reading from (1-indexed). If None, starts from beginning.",
    )
    line_count: int | None = Field(
        default=None,
        description="Number of lines to read. If None, reads to end of file.",
    )


class FileReadTool(BaseEnvironmentTool):
    """Read contents of text files with optional line ranges.

    Use this tool to:
    - Read configuration files, source code, logs
    - Inspect file contents before making decisions
    - Load reference documentation or data files

    Supports reading entire files or specific line ranges for efficiency.
    """

    name: str = "read_file"
    description: str = """Read the contents of a text file.

Use this to read configuration files, source code, logs, or any text file.
You can optionally specify start_line and line_count to read specific portions.

Examples:
- Read entire file: path="config.yaml"
- Read lines 100-149: path="large.log", start_line=100, line_count=50
"""
    args_schema: type[BaseModel] = FileReadInput

    def _run(
        self,
        path: str,
        start_line: int | None = None,
        line_count: int | None = None,
    ) -> str:
        """Read file contents with optional line range.

        Args:
            path: Path to the file to read.
            start_line: Line to start reading from (1-indexed).
            line_count: Number of lines to read.

        Returns:
            File contents with metadata header, or error message.
        """
        # Validate path against allowed_paths
        valid, result = self._validate_path(path)
        if not valid:
            return f"Error: {result}"

        assert isinstance(result, Path)  # noqa: S101
        file_path = result

        # Check file exists and is a file
        if not file_path.exists():
            return f"Error: File not found: {path}"

        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if start_line is None and line_count is None:
                    # Read entire file
                    content = f.read()
                else:
                    # Read specific line range
                    lines = f.readlines()
                    start_idx = (start_line or 1) - 1  # Convert to 0-indexed
                    start_idx = max(0, start_idx)  # Ensure non-negative

                    if line_count is not None:
                        end_idx = start_idx + line_count
                    else:
                        end_idx = len(lines)

                    content = "".join(lines[start_idx:end_idx])

            # Get file metadata
            stat = file_path.stat()
            total_lines = content.count("\n") + (
                1 if content and not content.endswith("\n") else 0
            )

            # Format output with metadata header
            header = f"File: {path}\n"
            header += f"Size: {self._format_size(stat.st_size)} | Lines: {total_lines}"

            if start_line is not None or line_count is not None:
                header += (
                    f" | Range: {start_line or 1}-{(start_line or 1) + total_lines - 1}"
                )

            header += "\n" + "=" * 60 + "\n"

            return header + content

        except UnicodeDecodeError:
            return f"Error: File is not a text file or has encoding issues: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"
