"""Regression tests for path-traversal-safe archive extraction.

Covers the fallback used on Python versions without tarfile extraction filters.
`tarfile.extractall(..., filter="data")` handles links safely; the fallback
refuses link members outright and checks every other member's path.
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
        with pytest.raises(ValueError, match="Blocked link member"):
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
        with pytest.raises(ValueError, match="Blocked link member"):
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
        with pytest.raises(ValueError, match="Blocked link member"):
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


def test_refuses_links_with_upgrade_hint(tmp_path: Path) -> None:
    """Even a link that stays inside dest is refused; filter="data" handles it."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        payload = b"hi"
        info = tarfile.TarInfo("real.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("alias.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "real.txt"
        tf.addfile(link)

    with _tar_from_members(build) as tf:
        with pytest.raises(
            ValueError, match=r"Python 3\.10\.12\+, 3\.11\.4\+, or 3\.12\+"
        ):
            _safe_extractall(tf, dest)

    assert list(dest.iterdir()) == []


def test_blocks_member_written_through_inside_symlink(tmp_path: Path) -> None:
    """A dest-relative symlink followed by a `..` member must not escape dest."""
    dest = tmp_path / "skill" / "dest"
    dest.mkdir(parents=True)

    def build(tf: tarfile.TarFile) -> None:
        link = tarfile.TarInfo("sub/up")
        link.type = tarfile.SYMTYPE
        link.linkname = ".."
        tf.addfile(link)
        payload = b"pwned"
        info = tarfile.TarInfo("sub/up/up/../evil.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="Blocked link member 'sub/up'"):
            _safe_extractall(tf, dest)

    assert not (tmp_path / "skill" / "evil.txt").exists()


def test_blocks_hardlink_to_dangling_symlink(tmp_path: Path) -> None:
    """tarfile re-creates a hardlink to a missing symlink member as that symlink,
    re-anchored at the hardlink's path; the fallback must not get that far."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        dangling = tarfile.TarInfo("a/b/link")
        dangling.type = tarfile.SYMTYPE
        dangling.linkname = "../.."
        tf.addfile(dangling)
        hard = tarfile.TarInfo("escape")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "a/b/link"
        tf.addfile(hard)
        payload = b"pwned"
        info = tarfile.TarInfo("escape/evil.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="Blocked link member"):
            _safe_extractall(tf, dest)

    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/tmp/evil.txt"])
def test_blocks_member_name_outside_destination(tmp_path: Path, name: str) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        payload = b"pwned"
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="Blocked path traversal attempt"):
            _safe_extractall(tf, dest)

    assert not (tmp_path / "evil.txt").exists()


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
