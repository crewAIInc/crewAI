"""Tool for listing directory contents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.experimental.environment_tools.base_environment_tool import (
    BaseEnvironmentTool,
)


class ListDirInput(BaseModel):
    """Input schema for listing directories."""

    path: str = Field(default=".", description="Directory path to list")
    pattern: str | None = Field(
        default=None,
        description="Glob pattern to filter entries (e.g., '*.py', '*.md')",
    )
    recursive: bool = Field(
        default=False,
        description="If True, list contents recursively including subdirectories",
    )


class ListDirTool(BaseEnvironmentTool):
    """List contents of a directory with optional filtering.

    Use this tool to:
    - Explore project structure
    - Find specific file types
    - Check what files exist in a directory
    - Navigate the file system
    """

    name: str = "list_directory"
    description: str = """List contents of a directory.

Use this to explore directories and find files. You can filter by pattern
and optionally list recursively.

Examples:
- List current dir: path="."
- List src folder: path="src/"
- Find Python files: path=".", pattern="*.py"
- Recursive listing: path="src/", recursive=True
"""
    args_schema: type[BaseModel] = ListDirInput

    def _run(
        self,
        path: str = ".",
        pattern: str | None = None,
        recursive: bool = False,
    ) -> str:
        """List directory contents.

        Args:
            path: Directory path to list.
            pattern: Glob pattern to filter entries.
            recursive: Whether to list recursively.

        Returns:
            Formatted directory listing or error message.
        """
        # Validate path against allowed_paths
        valid, result = self._validate_path(path)
        if not valid:
            return f"Error: {result}"

        dir_path = result

        # Check directory exists
        if not dir_path.exists():
            return f"Error: Directory not found: {path}"

        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            # Get entries based on pattern and recursive flag
            if pattern:
                if recursive:
                    entries = list(dir_path.rglob(pattern))
                else:
                    entries = list(dir_path.glob(pattern))
            else:
                if recursive:
                    entries = list(dir_path.rglob("*"))
                else:
                    entries = list(dir_path.iterdir())

            # Filter out hidden files (starting with .)
            entries = [e for e in entries if not e.name.startswith(".")]

            # Sort: directories first, then files, alphabetically
            entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            if not entries:
                if pattern:
                    return f"No entries matching '{pattern}' in {path}"
                return f"Directory is empty: {path}"

            # Format output
            result_lines = [f"Contents of {path}:"]
            result_lines.append("=" * 60)

            dirs = []
            files = []

            for entry in entries:
                try:
                    # Get relative path for recursive listings
                    if recursive:
                        display_name = str(entry.relative_to(dir_path))
                    else:
                        display_name = entry.name

                    if entry.is_dir():
                        dirs.append(f"📁 {display_name}/")
                    else:
                        size = entry.stat().st_size
                        size_str = self._format_size(size)
                        files.append(f"📄 {display_name} ({size_str})")
                except (OSError, PermissionError):
                    # Skip entries we can't access
                    continue

            # Output directories first, then files
            if dirs:
                result_lines.extend(dirs)
            if files:
                if dirs:
                    result_lines.append("")  # Blank line between dirs and files
                result_lines.extend(files)

            result_lines.append("")
            result_lines.append(f"Total: {len(dirs)} directories, {len(files)} files")

            return "\n".join(result_lines)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error listing directory: {e}"
