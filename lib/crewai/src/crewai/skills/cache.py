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
import zipfile

from crewai.skills.validation import versions_match


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

    def get_cached_path(
        self, org: str, name: str, version: str | None = None
    ) -> Path | None:
        """Return the cached skill directory path if usable, else None.

        Args:
            org: Organisation slug.
            name: Skill name.
            version: When given, the cached entry must record this version.
                The cache holds one version per skill, so a pinned lookup for a
                different version reports a miss and the caller re-downloads
                rather than loading the wrong version.

        Returns:
            The cached skill directory, or None on a miss.
        """
        skill_dir = self._skill_dir(org, name)
        meta_file = skill_dir / _META_FILENAME
        if not (skill_dir.is_dir() and meta_file.exists()):
            return None
        if version is not None and not self._records_version(meta_file, version):
            return None
        return skill_dir

    def _records_version(self, meta_file: Path, version: str) -> bool:
        """Return True when the cache metadata records *version*."""
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # ValueError covers both JSONDecodeError and the UnicodeDecodeError
            # a non-UTF-8 file raises, so a corrupted entry reads as a miss.
            _logger.debug("Unreadable cache entry: %s", meta_file, exc_info=True)
            return False
        if not isinstance(meta, dict):
            _logger.debug("Malformed cache entry: %s", meta_file)
            return False
        # versions_match() treats a non-string version as no match, so a
        # corrupted entry reads as a miss rather than raising.
        return versions_match(version, meta.get("version"))

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
        if skill_dir.exists():
            import shutil

            shutil.rmtree(skill_dir)
        skill_dir.mkdir(parents=True, exist_ok=True)

        import io

        try:
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
                try:
                    tf.extractall(skill_dir, filter="data")
                except TypeError:
                    _safe_extractall(tf, skill_dir)
        except tarfile.TarError:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                _safe_extract_zip(zf, skill_dir)

        meta: SkillMetadata = {
            "org": org,
            "name": name,
            "version": version,
            "installed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        (skill_dir / _META_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
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
                        results.append(
                            json.loads(meta_file.read_text(encoding="utf-8"))
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
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
    """Path-traversal-safe extraction for Python versions without tar filters.

    Only regular files and directories are extracted. Symlink and hardlink
    members are refused outright: they can redirect a later member's write
    outside ``dest`` in ways a per-member check cannot see (a member written
    through an earlier symlink, or a hardlink that ``tarfile`` re-creates as
    the dangling symlink it names). ``filter="data"`` handles links safely and
    is used whenever the running Python provides it. Mirrors
    ``crewai_cli.skills.main._safe_extractall``; change both together.
    """
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(
                f"Blocked link member {member.name!r}: installing skills that "
                "contain links needs Python 3.10.12+, 3.11.4+, or 3.12+"
            )
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Blocked unsupported tar member: {member.name!r}")
        if not (dest / member.name).resolve().is_relative_to(dest_resolved):
            raise ValueError(f"Blocked path traversal attempt: {member.name!r}")
    tf.extractall(dest)  # noqa: S202 - every member checked above, and no links


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Path-traversal-safe ZIP extraction."""
    dest_resolved = dest.resolve()
    for member in zf.namelist():
        member_path = (dest / member).resolve()
        if not member_path.is_relative_to(dest_resolved):
            raise ValueError(f"Blocked path traversal attempt: {member!r}")
    zf.extractall(dest)  # noqa: S202
