"""Tests for SkillCacheManager."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from crewai.skills.cache import SkillCacheManager, _safe_extractall


def _make_tar_gz(files: dict[str, str]) -> bytes:
    """Build an in-memory .tar.gz containing the given filename → content mapping."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz_buf = io.BytesIO()
        with tarfile.open(fileobj=gz_buf, mode="w") as tf:
            for name, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        gz.write(gz_buf.getvalue())
    buf.seek(0)
    # Re-create properly: gzip wrapping a tar stream
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tf:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _tar_from_members(build) -> tarfile.TarFile:
    """Build an in-memory tar archive via `build(tf)` and return it for reading."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        build(tf)
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r")


class TestSkillCacheManager:
    def test_get_cached_path_missing(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        assert cache.get_cached_path("acme", "my-skill") is None

    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        archive = _make_tar_gz({"SKILL.md": "---\nname: my-skill\n---\nHello"})
        dest = cache.store("acme", "my-skill", "1.0.0", archive)

        assert dest.is_dir()
        assert (dest / "SKILL.md").exists()

        retrieved = cache.get_cached_path("acme", "my-skill")
        assert retrieved == dest

    def test_store_writes_metadata(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        archive = _make_tar_gz({"SKILL.md": "content"})
        dest = cache.store("acme", "my-skill", "2.3.4", archive)

        meta_file = dest / ".crewai_meta.json"
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text())
        assert meta["org"] == "acme"
        assert meta["name"] == "my-skill"
        assert meta["version"] == "2.3.4"
        assert "installed_at" in meta

    def test_store_overwrites_previous_version(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        archive_v1 = _make_tar_gz({"SKILL.md": "v1", "extra.txt": "old"})
        cache.store("acme", "my-skill", "1.0.0", archive_v1)

        archive_v2 = _make_tar_gz({"SKILL.md": "v2"})
        dest = cache.store("acme", "my-skill", "2.0.0", archive_v2)

        assert not (dest / "extra.txt").exists()
        assert (dest / "SKILL.md").read_text() == "v2"

        meta = json.loads((dest / ".crewai_meta.json").read_text())
        assert meta["version"] == "2.0.0"

    def test_list_cached_empty(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        assert cache.list_cached() == []

    def test_list_cached(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        archive = _make_tar_gz({"SKILL.md": "x"})
        cache.store("acme", "skill-a", "1.0.0", archive)
        cache.store("acme", "skill-b", "0.1.0", archive)
        cache.store("other-org", "skill-c", None, archive)

        entries = cache.list_cached()
        names = {e["name"] for e in entries}
        assert names == {"skill-a", "skill-b", "skill-c"}

    def test_invalidate_existing(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        archive = _make_tar_gz({"SKILL.md": "x"})
        cache.store("acme", "my-skill", "1.0.0", archive)

        removed = cache.invalidate("acme", "my-skill")
        assert removed is True
        assert cache.get_cached_path("acme", "my-skill") is None

    def test_invalidate_missing(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        removed = cache.invalidate("acme", "ghost-skill")
        assert removed is False

    def test_store_version_none(self, tmp_path: Path) -> None:
        cache = SkillCacheManager(cache_root=tmp_path)
        archive = _make_tar_gz({"SKILL.md": "x"})
        dest = cache.store("acme", "my-skill", None, archive)
        meta = json.loads((dest / ".crewai_meta.json").read_text())
        assert meta["version"] is None


def test_safe_extractall_blocks_symlink_escaping_cache_destination(
    tmp_path: Path,
) -> None:
    """A symlink whose target escapes dest is rejected before extraction."""
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = str(outside)
        tf.addfile(link)
        payload = b"pwned"
        info = tarfile.TarInfo("link/evil.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="escaping destination"):
            _safe_extractall(tf, dest)

    assert not (outside / "evil.txt").exists()


def test_safe_extractall_blocks_hardlink_escaping_cache_destination(
    tmp_path: Path,
) -> None:
    """A hardlink whose target escapes dest is rejected."""
    dest = tmp_path / "dest"
    dest.mkdir()

    def build(tf: tarfile.TarFile) -> None:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.LNKTYPE
        link.linkname = "../outside.txt"
        tf.addfile(link)

    with _tar_from_members(build) as tf:
        with pytest.raises(ValueError, match="escaping destination"):
            _safe_extractall(tf, dest)


def test_safe_extractall_blocks_special_cache_tar_member(tmp_path: Path) -> None:
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


def test_safe_extractall_allows_benign_cache_symlink(tmp_path: Path) -> None:
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
        link.linkname = "real.txt"
        tf.addfile(link)

    with _tar_from_members(build) as tf:
        _safe_extractall(tf, dest)

    assert (dest / "real.txt").read_bytes() == b"hi"
    assert (dest / "alias.txt").is_symlink()
    assert (dest / "alias.txt").readlink() == Path("real.txt")
