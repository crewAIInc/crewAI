"""Provider text limits reach validation and chunk mode."""

from __future__ import annotations

import sys
from types import ModuleType

from crewai_files import FileBytes, TextFile
from crewai_files.processing.constraints import ProviderConstraints, TextConstraints
from crewai_files.processing.exceptions import FileTooLargeError
from crewai_files.processing.processor import FileProcessor
from crewai_files.processing.validators import validate_text
import pytest


def _text_file(
    monkeypatch: pytest.MonkeyPatch, data: bytes, mode: str = "auto"
) -> TextFile:
    """Build a text file without system libmagic."""
    magic_module = ModuleType("magic")

    def fake_from_buffer(*args: object, **kwargs: object) -> str:
        return "text/plain"

    magic_module.from_buffer = fake_from_buffer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "magic", magic_module)
    return TextFile(source=FileBytes(data=data, filename="note.txt"), mode=mode)


def test_validate_text_uses_the_provider_text_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider's text constraint is the size limit, not the unset general one."""
    constraints = ProviderConstraints(
        name="test", text=TextConstraints(max_size_bytes=5)
    )
    file = _text_file(monkeypatch, b"Hello, World!")

    with pytest.raises(FileTooLargeError):
        validate_text(file, constraints)


def test_chunk_mode_splits_text_by_the_provider_text_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk mode splits a text file that exceeds the provider's text limit."""
    constraints = ProviderConstraints(
        name="test", text=TextConstraints(max_size_bytes=300)
    )
    file = _text_file(monkeypatch, b"a" * 500, mode="chunk")

    processed = FileProcessor(constraints=constraints).process(file)

    assert isinstance(processed, list)
    assert [len(chunk.read()) for chunk in processed] == [300, 300, 300]
