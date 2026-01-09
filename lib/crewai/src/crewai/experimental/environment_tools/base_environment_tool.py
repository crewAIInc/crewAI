"""Base class for environment tools with path security."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from crewai.tools.base_tool import BaseTool


class BaseEnvironmentTool(BaseTool):
    """Base class for environment/file system tools with path security.

    Provides path validation to restrict file operations to allowed directories.
    This prevents path traversal attacks and enforces security sandboxing.

    Attributes:
        allowed_paths: List of paths that operations are restricted to.
            Empty list means allow all paths (no restrictions).
    """

    allowed_paths: list[str] = Field(
        default_factory=list,
        description="Restrict operations to these paths. Empty list allows all.",
    )

    def _validate_path(self, path: str) -> tuple[bool, Path | str]:
        """Validate and resolve a path against allowed_paths whitelist.

        Args:
            path: The path to validate.

        Returns:
            A tuple of (is_valid, result) where:
            - If valid: (True, resolved_path as Path)
            - If invalid: (False, error_message as str)
        """
        try:
            resolved = Path(path).resolve()

            # If no restrictions, allow all paths
            if not self.allowed_paths:
                return True, resolved

            # Check if path is within any allowed path
            for allowed in self.allowed_paths:
                allowed_resolved = Path(allowed).resolve()
                try:
                    # This will raise ValueError if resolved is not relative to allowed_resolved
                    resolved.relative_to(allowed_resolved)
                    return True, resolved
                except ValueError:
                    continue

            return (
                False,
                f"Path '{path}' is outside allowed paths: {self.allowed_paths}",
            )

        except Exception as e:
            return False, f"Invalid path '{path}': {e}"

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format.

        Args:
            size: Size in bytes.

        Returns:
            Human-readable size string (e.g., "1.5KB", "2.3MB").
        """
        if size < 1024:
            return f"{size}B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f}KB"
        if size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f}MB"
        return f"{size / (1024 * 1024 * 1024):.1f}GB"

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Subclasses must implement this method."""
        raise NotImplementedError("Subclasses must implement _run method")
