"""Tests for content-type detection fallbacks in core.sources."""

from pathlib import Path
import sys
from types import ModuleType

from crewai_files.core.sources import (
    detect_content_type,
    detect_content_type_from_path,
)
import pytest


class _BrokenMagic(ModuleType):
    """Stub standing in for python-magic with a broken libmagic.

    On Windows, ``python-magic`` probing %PATH% can load an incompatible
    MSYS/Cygwin libmagic (Git for Windows ships ``msys-magic-1.dll``), which
    crashes native CPython with an access violation or hangs the import.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.from_buffer_calls = 0

    def from_buffer(self, data: bytes, mime: bool = False) -> str:
        """Simulate the crash on the bytes path."""
        self.from_buffer_calls += 1
        raise OSError("exception: access violation reading libmagic buffer")

    def from_file(self, path: str, mime: bool = False) -> str:
        """Simulate the crash on the file path."""
        raise OSError("exception: access violation reading libmagic file")


@pytest.fixture
def broken_magic_on_windows(monkeypatch: pytest.MonkeyPatch) -> _BrokenMagic:
    """Install the broken python-magic stub and pretend we run on Windows."""
    stub = _BrokenMagic("magic")
    monkeypatch.setitem(sys.modules, "magic", stub)
    monkeypatch.setattr(sys, "platform", "win32")
    return stub


def test_detect_content_type_skips_magic_on_windows(
    broken_magic_on_windows: _BrokenMagic,
) -> None:
    """On Windows the broken libmagic probe must not be reached at all."""
    assert detect_content_type(b"Hello, World!") == "text/plain"
    assert broken_magic_on_windows.from_buffer_calls == 0


def test_detect_content_type_from_path_skips_magic_on_windows(
    broken_magic_on_windows: _BrokenMagic, tmp_path: Path
) -> None:
    """The file-path variant must not probe libmagic on Windows either."""
    file_path = tmp_path / "notes.txt"
    file_path.write_bytes(b"Hello, World!")
    assert detect_content_type_from_path(file_path) == "text/plain"
