"""Regression tests for path-traversal-safe archive extraction.

Guards against symlink/hardlink-based path traversal in the fallback used on
Python versions without tarfile extraction filters. The filtered path relies on
`tarfile.extractall(..., filter="data")`; the fallback must provide the same
protection by validating link targets, not just member names.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from crewai_cli.skills.main import _safe_extractall


def _tar_from_members(build) -> tarfile.TarFile:
    """Build an in-memory tar archive via `build(tf)` and return it for reading."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        build(tf)
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r")


def test_blocks_symlink_escaping_destination(tmp_path: Path) -> None:
    """A symlink whose target escapes dest, plus a file written through it,
    must be rejected before anything is extracted."""
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = str(outside)  # absolute path outside dest
        tf.addfile(link)
        payload = b"pwned"
        info = tarfile.TarInfo("link/evil.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="escaping destination"):
            _safe_extractall(tf, dest)

    assert not (outside / "evil.txt").exists()


def test_blocks_relative_symlink_escaping_destination(tmp_path: Path) -> None:
    """A relative symlink (../..) that escapes dest is also rejected."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        link = tarfile.TarInfo("sub/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"  # escapes dest from sub/
        tf.addfile(link)

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="escaping destination"):
            _safe_extractall(tf, dest)


def test_blocks_hardlink_escaping_destination(tmp_path: Path) -> None:
    """A hardlink whose target escapes dest is rejected."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.LNKTYPE
        link.linkname = "../outside.txt"  # escapes archive root
        tf.addfile(link)

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="escaping destination"):
            _safe_extractall(tf, dest)


def test_blocks_special_tar_member(tmp_path: Path) -> None:
    """Special tar members such as FIFOs are rejected."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        fifo = tarfile.TarInfo("pipe")
        fifo.type = tarfile.FIFOTYPE
        tf.addfile(fifo)

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="unsupported tar member"):
            _safe_extractall(tf, dest)


def test_allows_benign_relative_symlink(tmp_path: Path) -> None:
    """A symlink that stays within dest is permitted."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        payload = b"hi"
        info = tarfile.TarInfo("real.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("alias.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "real.txt"  # stays inside dest
        tf.addfile(link)

    with _tar_from_members(build) as tf:
        _safe_extractall(tf, dest)

    assert (dest / "real.txt").read_bytes() == b"hi"
    assert (dest / "alias.txt").is_symlink()
    assert (dest / "alias.txt").readlink() == Path("real.txt")


def test_allows_benign_archive(tmp_path: Path) -> None:
    """An ordinary archive of regular files extracts correctly."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        for name, body in (("SKILL.md", b"# skill"), ("scripts/run.py", b"print(1)")):
            payload = body
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

    with _tar_from_members(build) as tf:
        _safe_extractall(tf, dest)

    assert (dest / "SKILL.md").read_bytes() == b"# skill"
    assert (dest / "scripts" / "run.py").read_bytes() == b"print(1)"
