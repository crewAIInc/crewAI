"""Bedrock content block formatter."""

from __future__ import annotations

import base64
import re
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

# Converse API DocumentBlock.name only accepts alphanumerics, single
# whitespace characters, hyphens, parentheses and square brackets, and is
# limited to 200 characters.
_DOCUMENT_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9\s\-()\[\]]+")
_DOCUMENT_NAME_WHITESPACE = re.compile(r"\s+")
_DOCUMENT_NAME_MAX_LENGTH = 200


def _sanitize_document_name(name: str | None) -> str:
    """Make a file name acceptable as a Bedrock document block name.

    Characters the Converse API rejects (such as ``.`` or ``_`` in
    ``"q3_report.pdf"``) are replaced with spaces and whitespace runs are
    collapsed, so ``"q3_report.pdf"`` becomes ``"q3 report pdf"``.

    Args:
        name: Original file name.

    Returns:
        A valid document name, or ``"document"`` if nothing usable remains.
    """
    cleaned = _DOCUMENT_NAME_DISALLOWED.sub(" ", name or "")
    cleaned = _DOCUMENT_NAME_WHITESPACE.sub(" ", cleaned).strip()
    cleaned = cleaned[:_DOCUMENT_NAME_MAX_LENGTH].rstrip()
    return cleaned or "document"


def dedupe_document_names(blocks: list[dict[str, Any]]) -> None:
    """Give every Bedrock document block in a request a distinct name.

    Sanitizing can map different file names onto the same document name
    (``"a.pdf"`` and ``"a_pdf"`` both become ``"a pdf"``), and several
    unusable names fall back to ``"document"``. Bedrock rejects a request
    whose document blocks repeat a name, so later duplicates get a
    ``" (2)"``, ``" (3)"``, ... suffix while staying within the length limit.

    Args:
        blocks: Content blocks for one request; updated in place.
    """
    seen: set[str] = set()
    for block in blocks:
        document = block.get("document")
        if not isinstance(document, dict) or not isinstance(document.get("name"), str):
            continue
        base = document["name"]
        candidate = base
        counter = 2
        while candidate.casefold() in seen:
            suffix = f" ({counter})"
            trimmed = base[: _DOCUMENT_NAME_MAX_LENGTH - len(suffix)].rstrip()
            candidate = f"{trimmed}{suffix}"
            counter += 1
        seen.add(candidate.casefold())
        document["name"] = candidate


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
                    "name": _sanitize_document_name(name),
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
                    "name": _sanitize_document_name(name),
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
