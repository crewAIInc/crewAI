"""Tool for finding files by name pattern."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from crewai.experimental.environment_tools.base_environment_tool import (
    BaseEnvironmentTool,
)


class FileSearchInput(BaseModel):
    """Input schema for file search."""

    pattern: str = Field(
        ...,
        description="Filename pattern to search for (glob syntax, e.g., '*.py', 'test_*.py')",
    )
    path: str = Field(
        default=".",
        description="Directory to search in",
    )
    file_type: Literal["file", "dir", "all"] | None = Field(
        default="all",
        description="Filter by type: 'file' for files only, 'dir' for directories only, 'all' for both",
    )


class FileSearchTool(BaseEnvironmentTool):
    """Find files by name pattern.

    Use this tool to:
    - Find specific files in a codebase
    - Locate configuration files
    - Search for files matching a pattern
    """

    name: str = "find_files"
    description: str = """Find files by name pattern using glob syntax.

Searches recursively through directories to find matching files.

Examples:
- Find Python files: pattern="*.py", path="src/"
- Find test files: pattern="test_*.py"
- Find configs: pattern="*.yaml", path="."
- Find directories only: pattern="*", file_type="dir"
"""
    args_schema: type[BaseModel] = FileSearchInput

    def _run(
        self,
        pattern: str,
        path: str = ".",
        file_type: Literal["file", "dir", "all"] | None = "all",
    ) -> str:
        """Find files matching a pattern.

        Args:
            pattern: Glob pattern for filenames.
            path: Directory to search in.
            file_type: Filter by type ('file', 'dir', or 'all').

        Returns:
            List of matching files or error message.
        """
        # Validate path against allowed_paths
        valid, result = self._validate_path(path)
        if not valid:
            return f"Error: {result}"

        search_path = result

        # Check directory exists
        if not search_path.exists():
            return f"Error: Directory not found: {path}"

        if not search_path.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            # Find matching entries recursively
            matches = list(search_path.rglob(pattern))

            # Filter by type
            if file_type == "file":
                matches = [m for m in matches if m.is_file()]
            elif file_type == "dir":
                matches = [m for m in matches if m.is_dir()]

            # Filter out hidden files
            matches = [
                m for m in matches if not any(part.startswith(".") for part in m.parts)
            ]

            # Sort alphabetically
            matches.sort(key=lambda x: str(x).lower())

            if not matches:
                return f"No {file_type if file_type != 'all' else 'files'} matching '{pattern}' found in {path}"

            # Format output
            result_lines = [f"Found {len(matches)} matches for '{pattern}' in {path}:"]
            result_lines.append("=" * 60)

            for match in matches:
                # Get relative path from search directory
                rel_path = match.relative_to(search_path)

                if match.is_dir():
                    result_lines.append(f"📁 {rel_path}/")
                else:
                    try:
                        size = match.stat().st_size
                    except (OSError, PermissionError):
                        continue  # Skip files we can't stat
                    size_str = self._format_size(size)
                    result_lines.append(f"📄 {rel_path} ({size_str})")

            return "\n".join(result_lines)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error searching files: {e}"
