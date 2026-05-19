"""Cache manager for registry-downloaded skills.

Manages ~/.crewai/skills/{org}/{name}/ as the global skill cache.
One version is stored per skill (last install wins).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tarfile
from typing import TypedDict


_logger = logging.getLogger(__name__)


_CACHE_ROOT = Path.home() / ".crewai" / "skills"
_META_FILENAME = ".crewai_meta.json"


class SkillMetadata(TypedDict):
    org: str
    name: str
    version: str | None
    installed_at: str


class SkillCacheManager:
    """Manages the global skill cache at ~/.crewai/skills/."""

    def __init__(self, cache_root: Path | None = None) -> None:
        self._root = cache_root or _CACHE_ROOT

    def _skill_dir(self, org: str, name: str) -> Path:
        return self._root / org / name

    def get_cached_path(self, org: str, name: str) -> Path | None:
        """Return the cached skill directory path if it exists, else None."""
        skill_dir = self._skill_dir(org, name)
        meta_file = skill_dir / _META_FILENAME
        if skill_dir.is_dir() and meta_file.exists():
            return skill_dir
        return None

    def store(
        self, org: str, name: str, version: str | None, archive_bytes: bytes
    ) -> Path:
        """Unpack an archive into the cache and write metadata.

        Uses tarfile with filter='data' for path-traversal protection.

        Args:
            org: Organisation slug.
            name: Skill name.
            version: Semantic version string, or None if unknown.
            archive_bytes: Raw bytes of a .tar.gz archive.

        Returns:
            Path to the stored skill directory.
        """
        skill_dir = self._skill_dir(org, name)
        # Wipe any previous version
        if skill_dir.exists():
            import shutil

            shutil.rmtree(skill_dir)
        skill_dir.mkdir(parents=True, exist_ok=True)

        import io

        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
            try:
                tf.extractall(skill_dir, filter="data")
            except TypeError:
                # Python < 3.12 doesn't support filter= keyword; fall back safely
                _safe_extractall(tf, skill_dir)

        meta: SkillMetadata = {
            "org": org,
            "name": name,
            "version": version,
            "installed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        (skill_dir / _META_FILENAME).write_text(json.dumps(meta, indent=2))
        return skill_dir

    def list_cached(self) -> list[SkillMetadata]:
        """Return metadata for every cached skill."""
        results: list[SkillMetadata] = []
        if not self._root.exists():
            return results
        for org_dir in sorted(self._root.iterdir()):
            if not org_dir.is_dir():
                continue
            for skill_dir in sorted(org_dir.iterdir()):
                meta_file = skill_dir / _META_FILENAME
                if meta_file.exists():
                    try:
                        results.append(json.loads(meta_file.read_text()))
                    except (json.JSONDecodeError, KeyError):
                        _logger.debug(
                            "Skipping malformed cache entry: %s",
                            meta_file,
                            exc_info=True,
                        )
        return results

    def invalidate(self, org: str, name: str) -> bool:
        """Remove a cached skill.

        Returns:
            True if the cache entry existed and was removed, False otherwise.
        """
        skill_dir = self._skill_dir(org, name)
        if skill_dir.exists():
            import shutil

            shutil.rmtree(skill_dir)
            return True
        return False


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """Path-traversal-safe extraction for Python < 3.12."""
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        member_path = (dest / member.name).resolve()
        if not member_path.is_relative_to(dest_resolved):
            raise ValueError(f"Blocked path traversal attempt: {member.name!r}")
    tf.extractall(dest)  # noqa: S202
