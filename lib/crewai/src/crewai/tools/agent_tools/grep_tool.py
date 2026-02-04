"""Tool for searching file contents on disk using regex patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool


MAX_OUTPUT_CHARS = 50_000
MAX_FILES = 10_000
MAX_MATCHES_PER_FILE = 200
MAX_LINE_LENGTH = 500
BINARY_CHECK_SIZE = 8192

SKIP_DIRS = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
})


@dataclass
class MatchLine:
    """A single line from a search result."""

    line_number: int
    text: str
    is_match: bool  # True for match, False for context line


@dataclass
class FileSearchResult:
    """Search results for a single file."""

    file_path: Path
    matches: list[list[MatchLine]] = field(default_factory=list)
    match_count: int = 0


class GrepToolSchema(BaseModel):
    """Schema for grep tool arguments."""

    pattern: str = Field(..., description="Regex pattern to search for in file contents")
    path: str | None = Field(
        default=None,
        description="File or directory to search in. Defaults to current working directory.",
    )
    glob_pattern: str | None = Field(
        default=None,
        description="Glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}')",
    )
    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        default="content",
        description="Output mode: 'content' shows matching lines, 'files_with_matches' shows only file paths, 'count' shows match counts per file",
    )
    case_insensitive: bool = Field(
        default=False,
        description="Whether to perform case-insensitive matching",
    )
    context_lines: int = Field(
        default=0,
        description="Number of lines to show before and after each match",
    )
    include_line_numbers: bool = Field(
        default=True,
        description="Whether to prefix matching lines with line numbers",
    )


class GrepTool(BaseTool):
    """Tool for searching file contents on disk using regex patterns.

    Recursively searches files in a directory for lines matching a regex pattern.
    Supports glob filtering, context lines, and multiple output modes.
    """

    name: str = "grep"
    description: str = (
        "Search file contents on disk using regex patterns. "
        "Recursively searches files in a directory for matching lines. "
        "Returns matching content, file paths, or match counts."
    )
    args_schema: type[BaseModel] = GrepToolSchema

    def _run(
        self,
        pattern: str,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] = "content",
        case_insensitive: bool = False,
        context_lines: int = 0,
        include_line_numbers: bool = True,
        **kwargs: object,
    ) -> str:
        """Search files for a regex pattern.

        Args:
            pattern: Regex pattern to search for.
            path: File or directory to search. Defaults to cwd.
            glob_pattern: Glob pattern to filter files.
            output_mode: What to return.
            case_insensitive: Case-insensitive matching.
            context_lines: Lines of context around matches.
            include_line_numbers: Prefix lines with line numbers.

        Returns:
            Formatted search results as a string.
        """
        # Resolve search path
        search_path = Path(path) if path else Path(os.getcwd())
        if not search_path.exists():
            return f"Error: Path '{search_path}' does not exist."

        # Compile regex
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {e}"

        # Collect files
        files = self._collect_files(search_path, glob_pattern)

        # Search each file
        results: list[FileSearchResult] = []
        for file_path in files:
            result = self._search_file(file_path, compiled, context_lines)
            if result is not None:
                results.append(result)

        if not results:
            return "No matches found."

        # Format output
        if output_mode == "files_with_matches":
            output = self._format_files_with_matches(results)
        elif output_mode == "count":
            output = self._format_count(results)
        else:
            output = self._format_content(results, include_line_numbers)

        # Truncate if needed
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n\n... Output truncated. Try a narrower search pattern or glob filter."

        return output

    def _collect_files(self, search_path: Path, glob_pattern: str | None) -> list[Path]:
        """Collect files to search.

        Args:
            search_path: File or directory to search.
            glob_pattern: Optional glob pattern to filter files.

        Returns:
            List of file paths to search.
        """
        if search_path.is_file():
            return [search_path]

        pattern = glob_pattern or "*"
        files: list[Path] = []
        for p in search_path.rglob(pattern):
            if not p.is_file():
                continue
            # Skip hidden/build directories
            if any(part in SKIP_DIRS for part in p.relative_to(search_path).parts):
                continue
            files.append(p)
            if len(files) >= MAX_FILES:
                break

        return sorted(files)

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

    def _search_file(
        self,
        file_path: Path,
        compiled_pattern: re.Pattern[str],
        context_lines: int,
    ) -> FileSearchResult | None:
        """Search a single file for matches.

        Args:
            file_path: Path to the file.
            compiled_pattern: Compiled regex pattern.
            context_lines: Number of context lines around matches.

        Returns:
            FileSearchResult if matches found, None otherwise.
        """
        if self._is_binary_file(file_path):
            return None

        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (OSError, PermissionError):
            return None

        # Find matching line numbers
        match_line_nums: list[int] = []
        for i, line in enumerate(lines):
            if compiled_pattern.search(line):
                match_line_nums.append(i)
                if len(match_line_nums) >= MAX_MATCHES_PER_FILE:
                    break

        if not match_line_nums:
            return None

        # Build groups of contiguous match blocks with context
        groups: list[list[MatchLine]] = []
        current_group: list[MatchLine] = []
        prev_end = -1

        for match_idx in match_line_nums:
            start = max(0, match_idx - context_lines)
            end = min(len(lines), match_idx + context_lines + 1)

            # If this block doesn't overlap with the previous, start a new group
            if start > prev_end and current_group:
                groups.append(current_group)
                current_group = []

            for i in range(max(start, prev_end), end):
                text = lines[i].rstrip("\n\r")
                if len(text) > MAX_LINE_LENGTH:
                    text = text[:MAX_LINE_LENGTH] + "..."
                current_group.append(MatchLine(
                    line_number=i + 1,  # 1-indexed
                    text=text,
                    is_match=(i in match_line_nums),
                ))

            prev_end = end

        if current_group:
            groups.append(current_group)

        return FileSearchResult(
            file_path=file_path,
            matches=groups,
            match_count=len(match_line_nums),
        )

    def _format_content(
        self,
        results: list[FileSearchResult],
        include_line_numbers: bool,
    ) -> str:
        """Format results showing matching content.

        Args:
            results: List of file search results.
            include_line_numbers: Whether to include line numbers.

        Returns:
            Formatted string with file paths and matching lines.
        """
        parts: list[str] = []
        for result in results:
            parts.append(str(result.file_path))
            for group_idx, group in enumerate(result.matches):
                if group_idx > 0:
                    parts.append("--")
                for match_line in group:
                    if include_line_numbers:
                        parts.append(f"{match_line.line_number}: {match_line.text}")
                    else:
                        parts.append(match_line.text)
            parts.append("")  # blank line between files
        return "\n".join(parts).rstrip()

    def _format_files_with_matches(self, results: list[FileSearchResult]) -> str:
        """Format results showing only file paths.

        Args:
            results: List of file search results.

        Returns:
            One file path per line.
        """
        return "\n".join(str(r.file_path) for r in results)

    def _format_count(self, results: list[FileSearchResult]) -> str:
        """Format results showing match counts per file.

        Args:
            results: List of file search results.

        Returns:
            Filepath and count per line.
        """
        return "\n".join(f"{r.file_path}: {r.match_count}" for r in results)
