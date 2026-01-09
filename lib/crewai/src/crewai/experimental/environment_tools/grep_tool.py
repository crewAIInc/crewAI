"""Tool for searching patterns in files using grep."""

from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from crewai.experimental.environment_tools.base_environment_tool import (
    BaseEnvironmentTool,
)


class GrepInput(BaseModel):
    """Input schema for grep search."""

    pattern: str = Field(..., description="Search pattern (supports regex)")
    path: str = Field(..., description="File or directory to search in")
    recursive: bool = Field(
        default=True,
        description="Search recursively in directories",
    )
    ignore_case: bool = Field(
        default=False,
        description="Case-insensitive search",
    )
    context_lines: int = Field(
        default=2,
        description="Number of context lines to show before/after matches",
    )


class GrepTool(BaseEnvironmentTool):
    """Search for text patterns in files using grep.

    Use this tool to:
    - Find where a function or class is defined
    - Search for error messages in logs
    - Locate configuration values
    - Find TODO comments or specific patterns
    """

    name: str = "grep_search"
    description: str = """Search for text patterns in files using grep.

Supports regex patterns. Returns matching lines with context.

Examples:
- Find function: pattern="def process_data", path="src/"
- Search logs: pattern="ERROR", path="logs/app.log"
- Case-insensitive: pattern="todo", path=".", ignore_case=True
"""
    args_schema: type[BaseModel] = GrepInput

    def _run(
        self,
        pattern: str,
        path: str,
        recursive: bool = True,
        ignore_case: bool = False,
        context_lines: int = 2,
    ) -> str:
        """Search for patterns in files.

        Args:
            pattern: Search pattern (regex supported).
            path: File or directory to search in.
            recursive: Whether to search recursively.
            ignore_case: Whether to ignore case.
            context_lines: Lines of context around matches.

        Returns:
            Search results or error message.
        """
        # Validate path against allowed_paths
        valid, result = self._validate_path(path)
        if not valid:
            return f"Error: {result}"

        search_path = result

        # Check path exists
        if not search_path.exists():
            return f"Error: Path not found: {path}"

        try:
            # Build grep command safely
            cmd = ["grep", "--color=never"]

            # Add recursive flag if searching directory
            if recursive and search_path.is_dir():
                cmd.append("-r")

            # Case insensitive
            if ignore_case:
                cmd.append("-i")

            # Context lines
            if context_lines > 0:
                cmd.extend(["-C", str(context_lines)])

            # Show line numbers
            cmd.append("-n")

            # Use -- to prevent pattern from being interpreted as option
            cmd.append("--")
            cmd.append(pattern)
            cmd.append(str(search_path))

            # Execute with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Found matches
                output = result.stdout
                # Count actual match lines (not context lines)
                match_lines = [
                    line
                    for line in output.split("\n")
                    if line and not line.startswith("--")
                ]
                match_count = len(match_lines)

                header = f"Found {match_count} matches for '{pattern}' in {path}\n"
                header += "=" * 60 + "\n"
                return header + output

            if result.returncode == 1:
                # No matches found (grep returns 1 for no matches)
                return f"No matches found for '{pattern}' in {path}"

            # Error occurred
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error: {error_msg}"

        except subprocess.TimeoutExpired:
            return "Error: Search timed out (>30s). Try narrowing the search path."
        except FileNotFoundError:
            return (
                "Error: grep command not found. Ensure grep is installed on the system."
            )
        except Exception as e:
            return f"Error during search: {e}"
