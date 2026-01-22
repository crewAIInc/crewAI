"""OpenAI content block formatter."""

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


class OpenAIFormatter:
    """Formats resolved files into OpenAI content blocks."""

    def format_block(
        self,
        file: FileInput,
        resolved: ResolvedFile,
    ) -> dict[str, Any] | None:
        """Format a resolved file into an OpenAI content block.

        Args:
            file: Original file input with metadata.
            resolved: Resolved file.

        Returns:
            Content block dict or None if not supported.
        """
        content_type = file.content_type

        if isinstance(resolved, FileReference):
            return {
                "type": "file",
                "file": {"file_id": resolved.file_id},
            }

        if isinstance(resolved, UrlReference):
            return {
                "type": "image_url",
                "image_url": {"url": resolved.url},
            }

        if isinstance(resolved, InlineBase64):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{resolved.content_type};base64,{resolved.data}"
                },
            }

        data = base64.b64encode(file.read()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{data}"},
        }
