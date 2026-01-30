"""Bedrock content block formatter."""

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


_DOCUMENT_FORMATS: dict[str, str] = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/html": "html",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}

_VIDEO_FORMATS: dict[str, str] = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/x-matroska": "mkv",
    "video/webm": "webm",
    "video/x-flv": "flv",
    "video/mpeg": "mpeg",
    "video/3gpp": "three_gp",
}


class BedrockFormatter:
    """Formats resolved files into Bedrock Converse API content blocks."""

    def __init__(self, s3_bucket_owner: str | None = None) -> None:
        """Initialize formatter.

        Args:
            s3_bucket_owner: Optional S3 bucket owner for file references.
        """
        self.s3_bucket_owner = s3_bucket_owner

    def format_block(
        self,
        file: FileInput,
        resolved: ResolvedFileType,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        """Format a resolved file into a Bedrock content block.

        Args:
            file: Original file input with metadata.
            resolved: Resolved file.
            name: File name (required for document blocks).

        Returns:
            Content block dict or None if not supported.
        """
        content_type = file.content_type

        if isinstance(resolved, FileReference):
            if not resolved.file_uri:
                raise ValueError("Bedrock requires file_uri for FileReference (S3 URI)")
            return self._format_s3_block(content_type, resolved.file_uri, name)

        if isinstance(resolved, InlineBytes):
            return self._format_bytes_block(content_type, resolved.data, name)

        if isinstance(resolved, InlineBase64):
            file_bytes = base64.b64decode(resolved.data)
            return self._format_bytes_block(content_type, file_bytes, name)

        if isinstance(resolved, UrlReference):
            raise ValueError(
                "Bedrock does not support URL references - resolve to bytes first"
            )

        raise TypeError(f"Unexpected resolved type: {type(resolved).__name__}")

    def _format_s3_block(
        self,
        content_type: str,
        file_uri: str,
        name: str | None,
    ) -> dict[str, Any] | None:
        """Format block with S3 location source.

        Args:
            content_type: MIME type.
            file_uri: S3 URI.
            name: File name for documents.

        Returns:
            Content block dict or None.
        """
        s3_location: dict[str, Any] = {"uri": file_uri}
        if self.s3_bucket_owner:
            s3_location["bucketOwner"] = self.s3_bucket_owner

        if content_type.startswith("image/"):
            return {
                "image": {
                    "format": self._get_image_format(content_type),
                    "source": {"s3Location": s3_location},
                }
            }

        if content_type.startswith("video/"):
            video_format = _VIDEO_FORMATS.get(content_type)
            if video_format:
                return {
                    "video": {
                        "format": video_format,
                        "source": {"s3Location": s3_location},
                    }
                }
            return None

        doc_format = _DOCUMENT_FORMATS.get(content_type)
        if doc_format:
            return {
                "document": {
                    "name": name or "document",
                    "format": doc_format,
                    "source": {"s3Location": s3_location},
                }
            }

        return None

    def _format_bytes_block(
        self,
        content_type: str,
        file_bytes: bytes,
        name: str | None,
    ) -> dict[str, Any] | None:
        """Format block with inline bytes source.

        Args:
            content_type: MIME type.
            file_bytes: Raw file bytes.
            name: File name for documents.

        Returns:
            Content block dict or None.
        """
        if content_type.startswith("image/"):
            return {
                "image": {
                    "format": self._get_image_format(content_type),
                    "source": {"bytes": file_bytes},
                }
            }

        if content_type.startswith("video/"):
            video_format = _VIDEO_FORMATS.get(content_type)
            if video_format:
                return {
                    "video": {
                        "format": video_format,
                        "source": {"bytes": file_bytes},
                    }
                }
            return None

        doc_format = _DOCUMENT_FORMATS.get(content_type)
        if doc_format:
            return {
                "document": {
                    "name": name or "document",
                    "format": doc_format,
                    "source": {"bytes": file_bytes},
                }
            }

        return None

    @staticmethod
    def _get_image_format(content_type: str) -> str:
        """Get Bedrock image format from content type.

        Args:
            content_type: MIME type.

        Returns:
            Format string for Bedrock.
        """
        media_type = content_type.split("/")[-1]
        if media_type == "jpg":
            return "jpeg"
        return media_type
