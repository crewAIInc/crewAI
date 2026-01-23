"""OpenAI content block formatter."""

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


class OpenAIFormatter:
    """Formats resolved files into OpenAI content blocks."""

    @staticmethod
    def format_block(resolved: ResolvedFileType) -> dict[str, Any]:
        """Format a resolved file into an OpenAI content block.

        Args:
            resolved: Resolved file.

        Returns:
            Content block dict.

        Raises:
            TypeError: If resolved type is not supported.
        """
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

        if isinstance(resolved, InlineBytes):
            data = base64.b64encode(resolved.data).decode("ascii")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{resolved.content_type};base64,{data}"},
            }

        raise TypeError(f"Unexpected resolved type: {type(resolved).__name__}")
