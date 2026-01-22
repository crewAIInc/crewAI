"""Base formatter protocol for provider-specific content blocks."""

from __future__ import annotations

from typing import Any, Protocol

from crewai_files.core.resolved import ResolvedFile
from crewai_files.core.types import FileInput


class ContentFormatter(Protocol):
    """Protocol for formatting resolved files into provider content blocks."""

    def format_block(
        self,
        file: FileInput,
        resolved: ResolvedFile,
    ) -> dict[str, Any] | None:
        """Format a resolved file into a provider-specific content block.

        Args:
            file: Original file input with metadata.
            resolved: Resolved file (FileReference, InlineBase64, etc.).

        Returns:
            Content block dict or None if file type not supported.
        """
        ...
