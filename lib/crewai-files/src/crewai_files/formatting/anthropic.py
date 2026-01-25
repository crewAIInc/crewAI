"""Anthropic content block formatter."""

from __future__ import annotations

import base64
from typing import Any

from crewai_files.core.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFileType,
    UrlReference,
)
from crewai_files.core.types import FileInput


class AnthropicFormatter:
    """Formats resolved files into Anthropic content blocks."""

    def format_block(
        self,
        file: FileInput,
        resolved: ResolvedFileType,
    ) -> dict[str, Any] | None:
        """Format a resolved file into an Anthropic content block.

        Args:
            file: Original file input with metadata.
            resolved: Resolved file.

        Returns:
            Content block dict or None if not supported.
        """
        content_type = file.content_type
        block_type = self._get_block_type(content_type)
        if block_type is None:
            return None

        if isinstance(resolved, FileReference):
            return {
                "type": block_type,
                "source": {
                    "type": "file",
                    "file_id": resolved.file_id,
                },
                "cache_control": {"type": "ephemeral"},
            }

        if isinstance(resolved, UrlReference):
            return {
                "type": block_type,
                "source": {
                    "type": "url",
                    "url": resolved.url,
                },
                "cache_control": {"type": "ephemeral"},
            }

        if isinstance(resolved, InlineBase64):
            return {
                "type": block_type,
                "source": {
                    "type": "base64",
                    "media_type": resolved.content_type,
                    "data": resolved.data,
                },
                "cache_control": {"type": "ephemeral"},
            }

        if isinstance(resolved, InlineBytes):
            return {
                "type": block_type,
                "source": {
                    "type": "base64",
                    "media_type": resolved.content_type,
                    "data": base64.b64encode(resolved.data).decode("ascii"),
                },
                "cache_control": {"type": "ephemeral"},
            }

        raise TypeError(f"Unexpected resolved type: {type(resolved).__name__}")

    @staticmethod
    def _get_block_type(content_type: str) -> str | None:
        """Get Anthropic block type for content type.

        Args:
            content_type: MIME type.

        Returns:
            Block type string or None if not supported.
        """
        if content_type.startswith("image/"):
            return "image"
        if content_type == "application/pdf":
            return "document"
        return None
