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


class OpenAIResponsesFormatter:
    """Formats resolved files into OpenAI Responses API content blocks.

    The Responses API uses a different format than Chat Completions:
    - Text uses `type: "input_text"` instead of `type: "text"`
    - Images use `type: "input_image"` with `file_id` or `image_url`
    - PDFs use `type: "input_file"` with `file_id`, `file_url`, or `file_data`
    """

    @staticmethod
    def format_text_content(text: str) -> dict[str, Any]:
        """Format text as an OpenAI Responses API content block.

        Args:
            text: The text content to format.

        Returns:
            A content block with type "input_text".
        """
        return {"type": "input_text", "text": text}

    @staticmethod
    def format_block(resolved: ResolvedFileType, content_type: str) -> dict[str, Any]:
        """Format a resolved file into an OpenAI Responses API content block.

        Args:
            resolved: Resolved file.
            content_type: MIME type of the file.

        Returns:
            Content block dict.

        Raises:
            TypeError: If resolved type is not supported.
        """
        is_image = content_type.startswith("image/")
        is_pdf = content_type == "application/pdf"

        if isinstance(resolved, FileReference):
            if is_image:
                return {
                    "type": "input_image",
                    "file_id": resolved.file_id,
                }
            if is_pdf:
                return {
                    "type": "input_file",
                    "file_id": resolved.file_id,
                }
            raise TypeError(
                f"Unsupported content type for Responses API: {content_type}"
            )

        if isinstance(resolved, UrlReference):
            if is_image:
                return {
                    "type": "input_image",
                    "image_url": resolved.url,
                }
            if is_pdf:
                return {
                    "type": "input_file",
                    "file_url": resolved.url,
                }
            raise TypeError(
                f"Unsupported content type for Responses API: {content_type}"
            )

        if isinstance(resolved, InlineBase64):
            if is_image:
                return {
                    "type": "input_image",
                    "image_url": f"data:{resolved.content_type};base64,{resolved.data}",
                }
            if is_pdf:
                return {
                    "type": "input_file",
                    "filename": "document.pdf",
                    "file_data": f"data:{resolved.content_type};base64,{resolved.data}",
                }
            raise TypeError(
                f"Unsupported content type for Responses API: {content_type}"
            )

        if isinstance(resolved, InlineBytes):
            data = base64.b64encode(resolved.data).decode("ascii")
            if is_image:
                return {
                    "type": "input_image",
                    "image_url": f"data:{resolved.content_type};base64,{data}",
                }
            if is_pdf:
                return {
                    "type": "input_file",
                    "filename": "document.pdf",
                    "file_data": f"data:{resolved.content_type};base64,{data}",
                }
            raise TypeError(
                f"Unsupported content type for Responses API: {content_type}"
            )

        raise TypeError(f"Unexpected resolved type: {type(resolved).__name__}")


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
