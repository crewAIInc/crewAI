"""Tests for FileProcessor.aprocess_files handling of STRICT-mode failures."""

from __future__ import annotations

from typing import Any

from crewai_files import AudioFile, FileBytes, ImageFile
from crewai_files.processing.exceptions import UnsupportedFileTypeError
from crewai_files.processing.processor import FileProcessor
import pytest


MINIMAL_PNG = b"\x89PNG\r\n\x1a\n"  # magic header is all content-type detection needs


def _strict_unsupported_audio() -> AudioFile:
    # OpenAI completions constraints have no audio support, so an audio
    # file in strict mode fails validation.
    return AudioFile(
        source=FileBytes(data=b"ID3fake-audio-bytes", filename="clip.mp3"),
        mode="strict",
    )


def _valid_image() -> ImageFile:
    return ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="ok.png"))


def test_process_files_raises_for_strict_failure() -> None:
    processor = FileProcessor(constraints="openai")

    with pytest.raises(UnsupportedFileTypeError):
        processor.process_files({"bad": _strict_unsupported_audio()})


@pytest.mark.asyncio
async def test_aprocess_files_surfaces_strict_failure() -> None:
    """The async twin must surface a STRICT failure like the sync path.

    aprocess_files used to swallow the exception in gather(return_exceptions)
    and silently drop the file from the batch, while process_files raises.
    """

    processor = FileProcessor(constraints="openai")

    with pytest.raises(UnsupportedFileTypeError):
        await processor.aprocess_files({"bad": _strict_unsupported_audio()})


@pytest.mark.asyncio
async def test_aprocess_files_still_skips_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary per-file errors stay best-effort: logged and skipped."""

    processor = FileProcessor(constraints="openai")
    original_process = processor.process

    def flaky_process(file: Any) -> Any:
        if file.filename == "bad.png":
            raise RuntimeError("boom")
        return original_process(file)

    monkeypatch.setattr(processor, "process", flaky_process)

    result = await processor.aprocess_files(
        {"bad": _valid_image_named("bad.png"), "good": _valid_image_named("good.png")}
    )

    assert list(result) == ["good"]


def _valid_image_named(name: str) -> ImageFile:
    return ImageFile(source=FileBytes(data=MINIMAL_PNG, filename=name))
