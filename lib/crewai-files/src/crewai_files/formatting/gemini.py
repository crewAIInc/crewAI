"""Gemini content block formatter."""

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


class GeminiFormatter:
    """Formats resolved files into Gemini content blocks."""

    @staticmethod
    def format_block(resolved: ResolvedFileType) -> dict[str, Any]:
        """Format a resolved file into a Gemini content block.

        Args:
            resolved: Resolved file.

        Returns:
            Content block dict.

        Raises:
            TypeError: If resolved type is not supported.
        """
        if isinstance(resolved, FileReference):
            if not resolved.file_uri:
                raise ValueError("Gemini requires file_uri for FileReference")
            return {
                "fileData": {
                    "mimeType": resolved.content_type,
                    "fileUri": resolved.file_uri,
                }
            }

        if isinstance(resolved, UrlReference):
            return {
                "fileData": {
                    "mimeType": resolved.content_type,
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

        if isinstance(resolved, InlineBytes):
            return {
                "inlineData": {
                    "mimeType": resolved.content_type,
                    "data": base64.b64encode(resolved.data).decode("ascii"),
                }
            }

        raise TypeError(f"Unexpected resolved type: {type(resolved).__name__}")
