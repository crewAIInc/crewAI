"""Tests for SkillCacheManager."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

from crewai.skills.cache import SkillCacheManager


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

        # Old file should be gone
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
