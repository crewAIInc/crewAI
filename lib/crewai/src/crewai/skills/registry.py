"""Registry reference resolution for the Agent Skills standard.

Handles @org/skill-name references, local-first resolution, and downloads
via the CrewAI+ API with a global cache at ~/.crewai/skills/.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

from crewai.skills.cache import SkillCacheManager


_logger = logging.getLogger(__name__)


class SkillNotCachedError(Exception):
    """Raised when a registry skill is not cached and the environment is non-interactive."""

    def __init__(self, ref: str) -> None:
        super().__init__(
            f"Skill {ref!r} is not cached locally. "
            f"Run `crewai skill install {ref}` to install it first."
        )
        self.ref = ref


def is_registry_ref(value: Any) -> bool:
    """Return True if *value* looks like a registry reference (@org/name)."""
    return isinstance(value, str) and value.startswith("@")


def parse_registry_ref(ref: str) -> tuple[str, str]:
    """Parse '@org/skill-name' into (org, name).

    Args:
        ref: A registry reference, e.g. '@acme/my-skill'.

    Returns:
        A (org, name) tuple.

    Raises:
        ValueError: If the reference format is invalid.
    """
    if not ref.startswith("@"):
        raise ValueError(f"Registry reference must start with '@', got: {ref!r}")
    without_at = ref[1:]
    if without_at.count("/") != 1:
        raise ValueError(
            f"Registry reference must be in '@org/name' format, got: {ref!r}"
        )
    org, name = without_at.split("/", 1)
    if (
        not org
        or not name
        or org.startswith(".")
        or name.startswith(".")
        or "/" in org
        or "/" in name
    ):
        raise ValueError(
            f"Registry reference org and name must be single, non-empty path "
            f"segments (no '..' or leading dots), got: {ref!r}"
        )
    return org, name


def _is_noninteractive() -> bool:
    """Return True in CI or explicitly non-interactive environments."""
    import os

    return (
        os.environ.get("CI") == "1"
        or os.environ.get("CREWAI_NONINTERACTIVE") == "1"
        or not sys.stdin.isatty()
    )


def resolve_registry_ref(
    ref: str,
    source: Any = None,
) -> Skill:  # type: ignore[name-defined]  # noqa: F821
    """Resolve a registry reference to a Skill object.

    Resolution order:
    1. ./skills/{name}/ in the current working directory (project-local)
    2. ~/.crewai/skills/{org}/{name}/ (global cache)
    3. Download from registry (interactive only; raises SkillNotCachedError in CI)

    Args:
        ref: A registry reference, e.g. '@acme/my-skill'.
        source: Optional source object passed through to skill loaders (for events).

    Returns:
        A Skill loaded at INSTRUCTIONS disclosure level.

    Raises:
        SkillNotCachedError: When not cached and running in non-interactive mode.
    """
    from crewai.skills.loader import activate_skill
    from crewai.skills.parser import load_skill_metadata

    org, name = parse_registry_ref(ref)

    # 1. Project-local: ./skills/{name}/
    local_path = Path.cwd() / "skills" / name
    if local_path.is_dir() and (local_path / "SKILL.md").exists():
        try:
            skill = load_skill_metadata(local_path)
            return activate_skill(skill, source=source)
        except Exception:
            _logger.debug("Failed to load local skill at %s", local_path, exc_info=True)

    # 2. Global cache
    cache = SkillCacheManager()
    cached_path = cache.get_cached_path(org, name)
    if cached_path is not None and (cached_path / "SKILL.md").exists():
        try:
            skill = load_skill_metadata(cached_path)
            return activate_skill(skill, source=source)
        except Exception:
            _logger.debug(
                "Failed to load cached skill at %s", cached_path, exc_info=True
            )

    # 3. Download
    if _is_noninteractive():
        raise SkillNotCachedError(ref)

    return download_skill(org, name, source=source)


def download_skill(
    org: str,
    name: str,
    source: Any = None,
) -> Skill:  # type: ignore[name-defined]  # noqa: F821
    """Download a skill from the registry and store it in the cache.

    Args:
        org: Organisation slug.
        name: Skill name.
        source: Optional source for event emission.

    Returns:
        The downloaded Skill at INSTRUCTIONS level.
    """
    from crewai.skills.loader import activate_skill
    from crewai.skills.parser import load_skill_metadata

    ref = f"@{org}/{name}"

    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.skill_events import (
            SkillDownloadCompletedEvent,
            SkillDownloadStartedEvent,
        )

        _has_events = True
    except ImportError:
        _has_events = False

    if _has_events:
        crewai_event_bus.emit(
            source,
            event=SkillDownloadStartedEvent(
                registry_ref=ref,
            ),
        )

    try:
        from crewai_core.plus_api import PlusAPI

        api = PlusAPI()
        response = api.get_skill(org, name)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download skill {ref!r} from registry: {exc}"
        ) from exc

    import base64

    import httpx

    version = data.get("latest_version") or data.get("version")

    download_url = data.get("download_url")
    if download_url:
        dl_response = httpx.get(download_url, follow_redirects=True)
        dl_response.raise_for_status()
        archive_bytes = dl_response.content
    else:
        encoded = data.get("file", "")
        # Strip data URI prefix if present
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        archive_bytes = base64.b64decode(encoded)

    cache = SkillCacheManager()
    skill_dir = cache.store(org, name, version, archive_bytes)

    if _has_events:
        crewai_event_bus.emit(
            source,
            event=SkillDownloadCompletedEvent(
                registry_ref=ref,
                version=version,
                cache_path=skill_dir,
            ),
        )

    if not (skill_dir / "SKILL.md").exists():
        raise RuntimeError(
            f"Skill archive for {ref!r} downloaded but no SKILL.md found in {skill_dir}"
        )
    skill = load_skill_metadata(skill_dir)
    return activate_skill(skill, source=source)
