"""Async reads of synchronous streams, and the upload path that needs them."""

from __future__ import annotations

import io
import sys
from types import ModuleType

from crewai_files import FileReference, FileStream
from crewai_files.core.types import File
from crewai_files.resolution.resolver import FileResolver, FileResolverConfig
from crewai_files.uploaders import UploadResult
import pytest


def _stream_file(monkeypatch: pytest.MonkeyPatch, data: bytes = b"hello") -> File:
    """Build a file backed by a synchronous stream without system libmagic."""
    magic_module = ModuleType("magic")

    def fake_from_buffer(*args: object, **kwargs: object) -> str:
        return "text/plain"

    magic_module.from_buffer = fake_from_buffer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "magic", magic_module)
    return File(source=FileStream(stream=io.BytesIO(data), filename="note.txt"))


class _RecordingUploader:
    """A provider uploader that reads the file the way the real ones do."""

    def __init__(self) -> None:
        self.seen: list[bytes] = []

    async def aupload(self, file: File, purpose: str | None = None) -> UploadResult:
        self.seen.append(await file.aread())
        return UploadResult(
            file_id="file-1",
            provider="anthropic",
            content_type=file.content_type,
        )


@pytest.mark.asyncio
async def test_aread_reads_a_synchronous_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream-backed file can be read asynchronously, like every other source."""
    file = _stream_file(monkeypatch)

    assert await file.aread() == b"hello"


@pytest.mark.asyncio
async def test_async_upload_of_a_stream_reaches_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upload is performed instead of failing and falling back to inline."""
    monkeypatch.setattr("crewai_files.resolution.resolver.UPLOAD_MAX_RETRIES", 1)
    file = _stream_file(monkeypatch)
    uploader = _RecordingUploader()
    resolver = FileResolver(config=FileResolverConfig(prefer_upload=True))
    resolver._uploaders["anthropic"] = uploader  # type: ignore[assignment]

    resolved = await resolver.aresolve(file, "anthropic")

    assert isinstance(resolved, FileReference)
    assert uploader.seen == [b"hello"]
