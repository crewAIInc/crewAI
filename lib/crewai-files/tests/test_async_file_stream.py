"""Regression coverage for typed files backed by async-only streams."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path

import aiofiles
from crewai_files import (
    AudioFile,
    File,
    FileBytes,
    FileResolver,
    FileStream,
    ImageFile,
    InlineBase64,
    InlineBytes,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai_files.core.sources import AsyncFileStream, is_file_source
from crewai_files.core.types import BaseFile
from crewai_files.uploaders.factory import ProviderType
import pytest


class AsyncBuffer:
    """A cursor-based async stream that records underlying reads."""

    def __init__(self, content: bytes) -> None:
        self.buffer = BytesIO(content)
        self.read_count = 0

    async def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        await asyncio.sleep(0)
        return self.buffer.read(size)


@pytest.mark.parametrize(
    "file_type", [File, TextFile, ImageFile, PDFFile, AudioFile, VideoFile]
)
def test_typed_file_accepts_async_source(file_type: type[BaseFile]) -> None:
    source = AsyncFileStream(stream=AsyncBuffer(b"content"), filename="file.txt")

    assert is_file_source(source)
    file = file_type(source=source)
    assert file.source is source
    assert file.filename == "file.txt"


@pytest.fixture
def text_path(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_bytes("Notes from an async handle: café".encode())
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("provider", ["openai", "anthropic", "bedrock"])
async def test_resolve_aiofiles_source(
    text_path: Path, preloaded: bool, provider: ProviderType
) -> None:
    expected = "Notes from an async handle: café".encode()
    async with aiofiles.open(text_path, "rb") as handle:
        source = AsyncFileStream(stream=handle, filename=text_path.name)
        if preloaded:
            assert await source.aread() == expected

        resolved = await FileResolver().aresolve(TextFile(source=source), provider)

    if provider == "bedrock":
        assert isinstance(resolved, InlineBytes)
        content = resolved.data
    else:
        assert isinstance(resolved, InlineBase64)
        content = base64.b64decode(resolved.data)
    assert content == expected
    assert resolved.content_type == "text/plain"


@pytest.mark.asyncio
async def test_resolve_mixed_batch_retains_all_sources() -> None:
    files = {
        "async": TextFile(
            source=AsyncFileStream(stream=AsyncBuffer(b"async"), filename="async.txt")
        ),
        "stream": TextFile(
            source=FileStream(stream=BytesIO(b"stream"), filename="stream.txt")
        ),
        "bytes": TextFile(source=FileBytes(data=b"bytes", filename="bytes.txt")),
    }

    resolved = await FileResolver().aresolve_files(files, "openai")

    assert resolved.keys() == files.keys()
    for key, result in resolved.items():
        assert isinstance(result, InlineBase64)
        assert base64.b64decode(result.data) == key.encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"", b"shared content"])
async def test_resolve_shared_async_source_reads_once(content: bytes) -> None:
    stream = AsyncBuffer(content)
    source = AsyncFileStream(stream=stream, filename="shared.txt")
    files = {"first": TextFile(source=source), "second": TextFile(source=source)}

    resolved = await FileResolver().aresolve_files(files, "openai")

    assert resolved.keys() == files.keys()
    for result in resolved.values():
        assert isinstance(result, InlineBase64)
        assert base64.b64decode(result.data) == content
    assert stream.read_count == 1
