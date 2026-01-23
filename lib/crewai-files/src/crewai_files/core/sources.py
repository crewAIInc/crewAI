"""Base file class for handling file inputs in tasks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import inspect
import mimetypes
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Protocol, cast, runtime_checkable

import aiofiles
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    GetCoreSchemaHandler,
    PrivateAttr,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema
from typing_extensions import TypeIs

from crewai_files.core.constants import DEFAULT_MAX_FILE_SIZE_BYTES, MAGIC_BUFFER_SIZE


@runtime_checkable
class AsyncReadable(Protocol):
    """Protocol for async readable streams."""

    async def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the stream."""
        ...


class _AsyncReadableValidator:
    """Pydantic validator for AsyncReadable types."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: None, info_arg=False
            ),
        )

    @staticmethod
    def _validate(value: Any) -> AsyncReadable:
        if isinstance(value, AsyncReadable):
            return value
        raise ValueError("Expected an async readable object with async read() method")


ValidatedAsyncReadable = Annotated[AsyncReadable, _AsyncReadableValidator()]


def _fallback_content_type(filename: str | None) -> str:
    """Get content type from filename extension or return default."""
    if filename:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
    return "application/octet-stream"


def generate_filename(content_type: str) -> str:
    """Generate a UUID-based filename with extension from content type.

    Args:
        content_type: MIME type to derive extension from.

    Returns:
        Filename in format "{uuid}{ext}" where ext includes the dot.
    """
    import uuid

    ext = mimetypes.guess_extension(content_type) or ""
    return f"{uuid.uuid4()}{ext}"


def detect_content_type(data: bytes, filename: str | None = None) -> str:
    """Detect MIME type from file content.

    Uses python-magic if available for accurate content-based detection,
    falls back to mimetypes module using filename extension.

    Args:
        data: Raw bytes to analyze (only first 2048 bytes are used).
        filename: Optional filename for extension-based fallback.

    Returns:
        The detected MIME type.
    """
    try:
        import magic

        result: str = magic.from_buffer(data[:MAGIC_BUFFER_SIZE], mime=True)
        return result
    except ImportError:
        return _fallback_content_type(filename)


def detect_content_type_from_path(path: Path, filename: str | None = None) -> str:
    """Detect MIME type from file path.

    Uses python-magic's from_file() for accurate detection without reading
    the entire file into memory.

    Args:
        path: Path to the file.
        filename: Optional filename for extension-based fallback.

    Returns:
        The detected MIME type.
    """
    try:
        import magic

        result: str = magic.from_file(str(path), mime=True)
        return result
    except ImportError:
        return _fallback_content_type(filename or path.name)


class _BinaryIOValidator:
    """Pydantic validator for BinaryIO types."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
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
    max_size_bytes: int = Field(
        default=DEFAULT_MAX_FILE_SIZE_BYTES,
        exclude=True,
        description="Maximum file size in bytes.",
    )
    _content: bytes | None = PrivateAttr(default=None)
    _content_type: str = PrivateAttr()

    @model_validator(mode="after")
    def _validate_file_exists(self) -> FilePath:
        """Validate that the file exists, is secure, and within size limits."""
        from crewai_files.processing.exceptions import FileTooLargeError

        path_str = str(self.path)
        if ".." in path_str:
            raise ValueError(f"Path traversal not allowed: {self.path}")

        if self.path.is_symlink():
            resolved = self.path.resolve()
            cwd = Path.cwd().resolve()
            if not str(resolved).startswith(str(cwd)):
                raise ValueError(f"Symlink escapes allowed directory: {self.path}")

        if not self.path.exists():
            raise ValueError(f"File not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")

        actual_size = self.path.stat().st_size
        if actual_size > self.max_size_bytes:
            raise FileTooLargeError(
                f"File exceeds max size ({actual_size} > {self.max_size_bytes})",
                file_name=str(self.path),
                actual_size=actual_size,
                max_size=self.max_size_bytes,
            )

        self._content_type = detect_content_type_from_path(self.path, self.path.name)
        return self

    @property
    def filename(self) -> str:
        """Get the filename from the path."""
        return self.path.name

    @property
    def content_type(self) -> str:
        """Get the content type."""
        return self._content_type

    def read(self) -> bytes:
        """Read the file content from disk."""
        if self._content is None:
            self._content = self.path.read_bytes()
        return self._content

    async def aread(self) -> bytes:
        """Async read the file content from disk."""
        if self._content is None:
            async with aiofiles.open(self.path, "rb") as f:
                self._content = await f.read()
        return self._content

    def read_chunks(self, chunk_size: int = 65536) -> Iterator[bytes]:
        """Stream file content in chunks without loading entirely into memory.

        Args:
            chunk_size: Size of each chunk in bytes.

        Yields:
            Chunks of file content.
        """
        with open(self.path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk

    async def aread_chunks(self, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        """Async streaming for non-blocking I/O.

        Args:
            chunk_size: Size of each chunk in bytes.

        Yields:
            Chunks of file content.
        """
        async with aiofiles.open(self.path, "rb") as f:
            while chunk := await f.read(chunk_size):
                yield chunk


class FileBytes(BaseModel):
    """File created from raw bytes content."""

    data: bytes = Field(description="Raw bytes content of the file.")
    filename: str | None = Field(default=None, description="Optional filename.")
    _content_type: str = PrivateAttr()

    @model_validator(mode="after")
    def _detect_content_type(self) -> FileBytes:
        """Detect and cache content type from data."""
        self._content_type = detect_content_type(self.data, self.filename)
        return self

    @property
    def content_type(self) -> str:
        """Get the content type."""
        return self._content_type

    def read(self) -> bytes:
        """Return the bytes content."""
        return self.data

    async def aread(self) -> bytes:
        """Async return the bytes content (immediate, already in memory)."""
        return self.data

    def read_chunks(self, chunk_size: int = 65536) -> Iterator[bytes]:
        """Stream bytes content in chunks.

        Args:
            chunk_size: Size of each chunk in bytes.

        Yields:
            Chunks of bytes content.
        """
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i : i + chunk_size]

    async def aread_chunks(self, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        """Async streaming (immediate yield since already in memory).

        Args:
            chunk_size: Size of each chunk in bytes.

        Yields:
            Chunks of bytes content.
        """
        for chunk in self.read_chunks(chunk_size):
            yield chunk


class FileStream(BaseModel):
    """File loaded from a file-like stream."""

    stream: ValidatedBinaryIO = Field(description="Binary file stream.")
    filename: str | None = Field(default=None, description="Optional filename.")
    _content: bytes | None = PrivateAttr(default=None)
    _content_type: str = PrivateAttr()

    @model_validator(mode="after")
    def _initialize(self) -> FileStream:
        """Extract filename and detect content type."""
        if self.filename is None:
            name = getattr(self.stream, "name", None)
            if name is not None:
                self.filename = Path(name).name

        position = self.stream.tell()
        self.stream.seek(0)
        header = self.stream.read(MAGIC_BUFFER_SIZE)
        self.stream.seek(position)
        self._content_type = detect_content_type(header, self.filename)
        return self

    @property
    def content_type(self) -> str:
        """Get the content type."""
        return self._content_type

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

    def __enter__(self) -> FileStream:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager and close stream."""
        self.close()

    def read_chunks(self, chunk_size: int = 65536) -> Iterator[bytes]:
        """Stream from underlying stream in chunks.

        Args:
            chunk_size: Size of each chunk in bytes.

        Yields:
            Chunks of stream content.
        """
        position = self.stream.tell()
        self.stream.seek(0)
        try:
            while chunk := self.stream.read(chunk_size):
                yield chunk
        finally:
            self.stream.seek(position)


class AsyncFileStream(BaseModel):
    """File loaded from an async stream.

    Use for async file handles like aiofiles objects or aiohttp response bodies.
    This is an async-only type - use aread() instead of read().

    Attributes:
        stream: Async file-like object with async read() method.
        filename: Optional filename for the stream.
    """

    stream: ValidatedAsyncReadable = Field(
        description="Async file stream with async read() method."
    )
    filename: str | None = Field(default=None, description="Optional filename.")
    _content: bytes | None = PrivateAttr(default=None)
    _content_type: str | None = PrivateAttr(default=None)

    @property
    def content_type(self) -> str:
        """Get the content type from stream content (cached). Requires aread() first."""
        if self._content is None:
            raise RuntimeError("Call aread() first to load content")
        if self._content_type is None:
            self._content_type = detect_content_type(self._content, self.filename)
        return self._content_type

    async def aread(self) -> bytes:
        """Async read the stream content. Content is cached after first read."""
        if self._content is None:
            self._content = await self.stream.read()
        return self._content

    async def aclose(self) -> None:
        """Async close the underlying stream."""
        if hasattr(self.stream, "close"):
            result = self.stream.close()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self) -> AsyncFileStream:
        """Async enter context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async exit context manager and close stream."""
        await self.aclose()

    async def aread_chunks(self, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        """Async stream content in chunks.

        Args:
            chunk_size: Size of each chunk in bytes.

        Yields:
            Chunks of stream content.
        """
        while chunk := await self.stream.read(chunk_size):
            yield chunk


class FileUrl(BaseModel):
    """File referenced by URL.

    For providers that support URL references, the URL is passed directly.
    For providers that don't, content is fetched on demand.

    Attributes:
        url: URL where the file can be accessed.
        filename: Optional filename (extracted from URL if not provided).
    """

    url: str = Field(description="URL where the file can be accessed.")
    filename: str | None = Field(default=None, description="Optional filename.")
    _content_type: str | None = PrivateAttr(default=None)
    _content: bytes | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_url(self) -> FileUrl:
        """Validate URL format."""
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme: {self.url}")
        return self

    @property
    def content_type(self) -> str:
        """Get the content type, guessing from URL extension if not set."""
        if self._content_type is None:
            self._content_type = self._guess_content_type()
        return self._content_type

    def _guess_content_type(self) -> str:
        """Guess content type from URL extension."""
        from urllib.parse import urlparse

        parsed = urlparse(self.url)
        path = parsed.path
        guessed, _ = mimetypes.guess_type(path)
        return guessed or "application/octet-stream"

    def read(self) -> bytes:
        """Fetch content from URL (for providers that don't support URL references)."""
        if self._content is None:
            import httpx

            response = httpx.get(self.url, follow_redirects=True)
            response.raise_for_status()
            self._content = response.content
            if "content-type" in response.headers:
                self._content_type = response.headers["content-type"].split(";")[0]
        return self._content

    async def aread(self) -> bytes:
        """Async fetch content from URL."""
        if self._content is None:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(self.url, follow_redirects=True)
                response.raise_for_status()
                self._content = response.content
                if "content-type" in response.headers:
                    self._content_type = response.headers["content-type"].split(";")[0]
        return self._content


FileSource = FilePath | FileBytes | FileStream | AsyncFileStream | FileUrl


def is_file_source(v: object) -> TypeIs[FileSource]:
    """Type guard to narrow input to FileSource."""
    return isinstance(v, (FilePath, FileBytes, FileStream, FileUrl))


def _normalize_source(value: Any) -> FileSource:
    """Convert raw input to appropriate source type."""
    if isinstance(value, (FilePath, FileBytes, FileStream, AsyncFileStream, FileUrl)):
        return value
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return FileUrl(url=value)
        return FilePath(path=Path(value))
    if isinstance(value, Path):
        return FilePath(path=value)
    if isinstance(value, bytes):
        return FileBytes(data=value)
    if isinstance(value, AsyncReadable):
        return AsyncFileStream(stream=value)
    if hasattr(value, "read") and hasattr(value, "seek"):
        return FileStream(stream=value)
    raise ValueError(f"Cannot convert {type(value).__name__} to file source")


RawFileInput = str | Path | bytes
FileSourceInput = Annotated[
    RawFileInput | FileSource, BeforeValidator(_normalize_source)
]
