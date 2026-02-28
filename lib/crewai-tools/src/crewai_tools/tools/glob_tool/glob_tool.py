"""Tool for finding files matching glob patterns."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


MAX_FILES = 1000
MAX_OUTPUT_CHARS = 30_000

SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }
)


@dataclass
class FileInfo:
    """Information about a matched file."""

    path: Path
    size: int
    is_dir: bool


class GlobToolSchema(BaseModel):
    """Schema for glob tool arguments."""

    pattern: str = Field(
        ...,
        description=(
            "Glob pattern to match files. Examples: '*.py' (Python files), "
            "'**/*.yaml' (all YAML files recursively), 'src/**/*.ts' (TypeScript in src), "
            "'test_*.py' (test files). Patterns not starting with '**/' are auto-prefixed for recursive search."
        ),
    )
    path: str | None = Field(
        default=None,
        description="Directory to search in. Defaults to current working directory.",
    )
    output_mode: Literal["paths", "tree", "detailed"] = Field(
        default="paths",
        description=(
            "Output format: 'paths' shows file paths one per line, "
            "'tree' shows directory tree structure, "
            "'detailed' includes file sizes."
        ),
    )
    include_hidden: bool = Field(
        default=False,
        description="Whether to include hidden files and directories (starting with '.').",
    )
    dirs_only: bool = Field(
        default=False,
        description="If True, only match directories, not files.",
    )
    files_only: bool = Field(
        default=True,
        description="If True (default), only match files, not directories.",
    )


class GlobTool(BaseTool):
    """Tool for finding files matching glob patterns.

    Recursively searches for files matching a glob pattern within a directory.
    Useful for discovering files by name, extension, or path pattern.
    Complements GrepTool which searches by file content.

    Example:
        >>> tool = GlobTool()
        >>> result = tool.run(pattern="*.py", path="/path/to/project")
        >>> result = tool.run(pattern="**/*.yaml", output_mode="detailed")
    """

    name: str = "glob"
    description: str = (
        "Find files matching a glob pattern. Use to discover files by name or extension. "
        "Examples: '*.py' finds all Python files, '**/*.yaml' finds YAML files recursively, "
        "'test_*.py' finds test files. Returns matching file paths sorted by modification time."
    )
    args_schema: type[BaseModel] = GlobToolSchema

    def _run(
        self,
        pattern: str,
        path: str | None = None,
        output_mode: Literal["paths", "tree", "detailed"] = "paths",
        include_hidden: bool = False,
        dirs_only: bool = False,
        files_only: bool = True,
        **kwargs: object,
    ) -> str:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern to match.
            path: Directory to search in. Defaults to cwd.
            output_mode: Output format (paths, tree, detailed).
            include_hidden: Whether to include hidden files.
            dirs_only: Only match directories.
            files_only: Only match files (default True).

        Returns:
            Formatted list of matching paths.
        """
        # Resolve search path
        search_path = Path(path) if path else Path(os.getcwd())
        if not search_path.exists():
            return f"Error: Path '{search_path}' does not exist."
        if not search_path.is_dir():
            return f"Error: Path '{search_path}' is not a directory."

        # Normalize pattern for recursive search
        normalized_pattern = pattern
        if not pattern.startswith("**/") and not pattern.startswith("/"):
            if "/" not in pattern:
                normalized_pattern = f"**/{pattern}"

        matches: list[FileInfo] = []
        try:
            for match_path in search_path.glob(normalized_pattern):
                if not include_hidden:
                    if any(
                        part.startswith(".")
                        for part in match_path.relative_to(search_path).parts
                    ):
                        continue

                rel_parts = match_path.relative_to(search_path).parts
                if any(part in SKIP_DIRS for part in rel_parts):
                    continue

                is_dir = match_path.is_dir()
                if dirs_only and not is_dir:
                    continue
                if files_only and is_dir:
                    continue

                try:
                    size = match_path.stat().st_size if not is_dir else 0
                    matches.append(FileInfo(path=match_path, size=size, is_dir=is_dir))
                except (OSError, PermissionError):
                    continue

                if len(matches) >= MAX_FILES:
                    break

        except Exception as e:
            return f"Error: Failed to search with pattern '{pattern}': {e!s}"

        if not matches:
            return f"No files found matching pattern '{pattern}' in {search_path}"

        try:
            matches.sort(key=lambda f: f.path.stat().st_mtime, reverse=True)
        except (OSError, PermissionError):
            matches.sort(key=lambda f: str(f.path))

        if output_mode == "detailed":
            output = self._format_detailed(matches, search_path)
        elif output_mode == "tree":
            output = self._format_tree(matches, search_path)
        else:
            output = self._format_paths(matches, search_path)

        summary = f"Found {len(matches)} file(s) matching '{pattern}'"
        if len(matches) >= MAX_FILES:
            summary += f" (limited to {MAX_FILES})"

        result = f"{summary}\n\n{output}"

        if len(result) > MAX_OUTPUT_CHARS:
            result = (
                result[:MAX_OUTPUT_CHARS]
                + "\n\n... Output truncated. Use a more specific pattern."
            )

        return result

    def _format_paths(self, matches: list[FileInfo], base_path: Path) -> str:
        """Format as simple list of paths."""
        return "\n".join(str(f.path) for f in matches)

    def _format_detailed(self, matches: list[FileInfo], base_path: Path) -> str:
        """Format with file sizes."""
        lines: list[str] = []
        for f in matches:
            size_str = self._format_size(f.size) if not f.is_dir else "<dir>"
            rel_path = (
                f.path.relative_to(base_path)
                if f.path.is_relative_to(base_path)
                else f.path
            )
            lines.append(f"{size_str:>10}  {rel_path}")
        return "\n".join(lines)

    def _format_tree(self, matches: list[FileInfo], base_path: Path) -> str:
        """Format as directory tree structure."""
        # Build tree structure
        tree: dict[str, list[str]] = {}
        for f in matches:
            try:
                rel_path = f.path.relative_to(base_path)
            except ValueError:
                rel_path = f.path

            parent = str(rel_path.parent) if rel_path.parent != Path(".") else "."
            if parent not in tree:
                tree[parent] = []
            tree[parent].append(rel_path.name + ("/" if f.is_dir else ""))

        # Format tree output
        lines: list[str] = [str(base_path)]
        for directory in sorted(tree.keys()):
            if directory != ".":
                lines.append(f"  {directory}/")
            for filename in sorted(tree[directory]):
                prefix = "    " if directory != "." else "  "
                lines.append(f"{prefix}{filename}")

        return "\n".join(lines)

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        size_float = float(size)
        for unit in ["B", "KB", "MB", "GB"]:
            if size_float < 1024:
                return (
                    f"{size_float:.0f}{unit}"
                    if unit == "B"
                    else f"{size_float:.1f}{unit}"
                )
            size_float /= 1024
        return f"{size_float:.1f}TB"
