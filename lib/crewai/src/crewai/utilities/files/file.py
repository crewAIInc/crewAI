"""Base file class for handling file inputs in tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast

import magic
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    GetCoreSchemaHandler,
    PrivateAttr,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema


def detect_content_type(data: bytes) -> str:
    """Detect MIME type from file content.

    Args:
        data: Raw bytes to analyze.

    Returns:
        The detected MIME type.
    """
    return magic.from_buffer(data, mime=True)


class _BinaryIOValidator:
    """Pydantic validator for BinaryIO types."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: None, info_arg=False
            ),
        )

    @staticmethod
    def _validate(value: Any) -> BinaryIO:
        if hasattr(value, "read") and hasattr(value, "seek"):
            return cast(BinaryIO, value)
        raise ValueError("Expected a binary file-like object with read() and seek()")


ValidatedBinaryIO = Annotated[BinaryIO, _BinaryIOValidator()]


class FilePath(BaseModel):
    """File loaded from a filesystem path."""

    path: Path = Field(description="Path to the file on the filesystem.")
    _content: bytes | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_file_exists(self) -> FilePath:
        """Validate that the file exists."""
        if not self.path.exists():
            raise ValueError(f"File not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")
        return self

    @property
    def filename(self) -> str:
        """Get the filename from the path."""
        return self.path.name

    @property
    def content_type(self) -> str:
        """Get the content type by reading file content."""
        return detect_content_type(self.read())

    def read(self) -> bytes:
        """Read the file content from disk."""
        if self._content is None:
            self._content = self.path.read_bytes()
        return self._content


class FileBytes(BaseModel):
    """File created from raw bytes content."""

    data: bytes = Field(description="Raw bytes content of the file.")
    filename: str | None = Field(default=None, description="Optional filename.")

    @property
    def content_type(self) -> str:
        """Get the content type from the data."""
        return detect_content_type(self.data)

    def read(self) -> bytes:
        """Return the bytes content."""
        return self.data


class FileStream(BaseModel):
    """File loaded from a file-like stream."""

    stream: ValidatedBinaryIO = Field(description="Binary file stream.")
    filename: str | None = Field(default=None, description="Optional filename.")
    _content: bytes | None = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        """Extract filename from stream if not provided."""
        if self.filename is None:
            name = getattr(self.stream, "name", None)
            if name is not None:
                object.__setattr__(self, "filename", Path(name).name)

    @property
    def content_type(self) -> str:
        """Get the content type from stream content."""
        return detect_content_type(self.read())

    def read(self) -> bytes:
        """Read the stream content. Content is cached after first read."""
        if self._content is None:
            position = self.stream.tell()
            self.stream.seek(0)
            self._content = self.stream.read()
            self.stream.seek(position)
        return self._content

    def close(self) -> None:
        """Close the underlying stream."""
        self.stream.close()


FileSource = FilePath | FileBytes | FileStream


def _normalize_source(value: Any) -> FileSource:
    """Convert raw input to appropriate source type."""
    if isinstance(value, (FilePath, FileBytes, FileStream)):
        return value
    if isinstance(value, Path):
        return FilePath(path=value)
    if isinstance(value, str):
        return FilePath(path=Path(value))
    if isinstance(value, bytes):
        return FileBytes(data=value)
    if hasattr(value, "read") and hasattr(value, "seek"):
        return FileStream(stream=value)
    raise ValueError(f"Cannot convert {type(value).__name__} to file source")


RawFileInput = str | Path | bytes
FileSourceInput = Annotated[
    RawFileInput | FileSource, BeforeValidator(_normalize_source)
]
