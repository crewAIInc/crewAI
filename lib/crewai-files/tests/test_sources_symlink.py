"""Tests for FilePath source validation, focusing on symlink containment."""

from __future__ import annotations

import os
from pathlib import Path

from crewai_files.core.sources import FilePath
import pytest


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _symlink_or_skip(target: str | Path, link: str | Path) -> None:
    """Create a symlink, skipping where the OS requires a privilege (Windows)."""
    try:
        os.symlink(str(target), str(link))
    except OSError as e:
        pytest.skip(f"symlink creation not permitted on this platform: {e}")


class TestFilePathSymlinkContainment:
    def test_symlink_into_working_directory_is_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allowed = tmp_path / "app"
        (allowed / "data").mkdir(parents=True)
        target = allowed / "data" / "input.png"
        target.write_bytes(PNG_MAGIC)
        link = allowed / "link.png"
        _symlink_or_skip(target, link)
        monkeypatch.chdir(allowed)

        assert FilePath(path=link).read() == PNG_MAGIC

    def test_symlink_to_unrelated_directory_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allowed = tmp_path / "app"
        allowed.mkdir()
        outside = tmp_path / "elsewhere" / "secret.png"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(PNG_MAGIC)
        link = allowed / "link.png"
        _symlink_or_skip(outside, link)
        monkeypatch.chdir(allowed)

        with pytest.raises(ValueError, match="Symlink escapes"):
            FilePath(path=link)

    def test_symlink_to_sibling_sharing_prefix_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "app-data" starts with the string "app" but is a different
        # directory; containment must compare path components, not
        # characters, or this escape is admitted.
        allowed = tmp_path / "app"
        allowed.mkdir()
        sibling = tmp_path / "app-data" / "secret.png"
        sibling.parent.mkdir(parents=True)
        sibling.write_bytes(PNG_MAGIC)
        link = allowed / "link.png"
        _symlink_or_skip(sibling, link)
        monkeypatch.chdir(allowed)

        with pytest.raises(ValueError, match="Symlink escapes"):
            FilePath(path=link)
