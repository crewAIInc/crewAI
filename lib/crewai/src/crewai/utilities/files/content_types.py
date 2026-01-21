"""Content-type specific file classes."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator

from crewai.utilities.files.file import (
    FileBytes,
    FilePath,
    FileSource,
    FileStream,
)


FileMode = Literal["strict", "auto", "warn", "chunk"]


ImageExtension = Literal[
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"
]
ImageContentType = Literal[
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/svg+xml",
]

PDFExtension = Literal[".pdf"]
PDFContentType = Literal["application/pdf"]

TextExtension = Literal[
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
]
TextContentType = Literal[
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/xml",
    "text/xml",
    "application/x-yaml",
    "text/yaml",
    "text/html",
]

AudioExtension = Literal[
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus"
]
AudioContentType = Literal[
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
    "audio/mp4",
    "audio/x-ms-wma",
    "audio/aiff",
    "audio/opus",
]

VideoExtension = Literal[
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"
]
VideoContentType = Literal[
    "video/mp4",
    "video/x-msvideo",
    "video/x-matroska",
    "video/quicktime",
    "video/webm",
    "video/x-flv",
    "video/x-ms-wmv",
    "video/mpeg",
]


class BaseFile(ABC, BaseModel):
    """Abstract base class for typed file wrappers.

    Provides common functionality for all file types including:
    - File source management
    - Content reading
    - Dict unpacking support (`**` syntax)
    - Per-file mode mode

    Can be unpacked with ** syntax: `{**ImageFile(source="./chart.png")}`
    which unpacks to: `{"chart": <ImageFile instance>}` using filename stem as key.

    Attributes:
        source: The underlying file source (path, bytes, or stream).
        mode: How to handle this file if it exceeds provider limits.
    """

    source: FileSource = Field(description="The underlying file source.")
    mode: FileMode = Field(
        default="auto",
        description="How to handle if file exceeds limits: strict, auto, warn, chunk.",
    )

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, v: str | Path | bytes | FileSource) -> FileSource:
        """Convert raw input to appropriate source type."""
        if isinstance(v, (FilePath, FileBytes, FileStream)):
            return v
        if isinstance(v, Path):
            return FilePath(path=v)
        if isinstance(v, str):
            return FilePath(path=Path(v))
        if isinstance(v, bytes):
            return FileBytes(data=v)
        if hasattr(v, "read") and hasattr(v, "seek"):
            return FileStream(stream=v)
        raise ValueError(f"Cannot convert {type(v).__name__} to file source")

    @property
    def filename(self) -> str | None:
        """Get the filename from the source."""
        return self.source.filename

    @property
    def content_type(self) -> str:
        """Get the content type from the source."""
        return self.source.content_type

    def read(self) -> bytes:
        """Read the file content as bytes."""
        return self.source.read()

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read the file content as string."""
        return self.read().decode(encoding)

    @property
    def _unpack_key(self) -> str:
        """Get the key to use when unpacking (filename stem)."""
        if self.source.filename:
            return Path(self.source.filename).stem
        return "file"

    def keys(self) -> list[str]:
        """Return keys for dict unpacking."""
        return [self._unpack_key]

    def __getitem__(self, key: str) -> Self:
        """Return self for dict unpacking."""
        if key == self._unpack_key:
            return self
        raise KeyError(key)


class ImageFile(BaseFile):
    """File representing an image.

    Supports common image formats: PNG, JPEG, GIF, WebP, BMP, TIFF, SVG.
    """


class PDFFile(BaseFile):
    """File representing a PDF document."""


class TextFile(BaseFile):
    """File representing a text document.

    Supports common text formats: TXT, MD, RST, CSV, JSON, XML, YAML, HTML.
    """


class AudioFile(BaseFile):
    """File representing an audio file.

    Supports common audio formats: MP3, WAV, OGG, FLAC, AAC, M4A, WMA.
    """


class VideoFile(BaseFile):
    """File representing a video file.

    Supports common video formats: MP4, AVI, MKV, MOV, WebM, FLV, WMV.
    """


class File(BaseFile):
    """Generic file that auto-detects the appropriate type.

    Use this when you don't want to specify the exact file type.
    The content type is automatically detected from the file contents.

    Example:
        >>> file = File(source="./document.pdf")
        >>> file = File(source="./image.png")
        >>> file = File(source=some_bytes)
    """
