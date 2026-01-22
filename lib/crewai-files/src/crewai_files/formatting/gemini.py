"""Gemini content block formatter."""

from __future__ import annotations

import base64
from typing import Any

from crewai_files.core.resolved import (
    FileReference,
    InlineBase64,
    ResolvedFile,
    UrlReference,
)
from crewai_files.core.types import FileInput


class GeminiFormatter:
    """Formats resolved files into Gemini content blocks."""

    def format_block(
        self,
        file: FileInput,
        resolved: ResolvedFile,
    ) -> dict[str, Any] | None:
        """Format a resolved file into a Gemini content block.

        Args:
            file: Original file input with metadata.
            resolved: Resolved file.

        Returns:
            Content block dict or None if not supported.
        """
        content_type = file.content_type

        if isinstance(resolved, FileReference) and resolved.file_uri:
            return {
                "fileData": {
                    "mimeType": resolved.content_type,
                    "fileUri": resolved.file_uri,
                }
            }

        if isinstance(resolved, UrlReference):
            return {
                "fileData": {
                    "mimeType": content_type,
                    "fileUri": resolved.url,
                }
            }

        if isinstance(resolved, InlineBase64):
            return {
                "inlineData": {
                    "mimeType": resolved.content_type,
                    "data": resolved.data,
                }
            }

        data = base64.b64encode(file.read()).decode("ascii")
        return {
            "inlineData": {
                "mimeType": content_type,
                "data": data,
            }
        }
