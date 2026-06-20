"""Experimental Skills Repository — registry refs, global cache, downloads.

This package contains the registry-backed pieces of the skills feature
(`@org/name` refs, `~/.crewai/skills/` cache, download events). The stable
filesystem-based skill loader still lives in `crewai.skills`.
"""

from crewai.experimental.skills.cache import SkillCacheManager
from crewai.experimental.skills.registry import (
    SkillNotCachedError,
    is_registry_ref,
    parse_registry_ref,
    resolve_registry_ref,
)


__all__ = [
    "SkillCacheManager",
    "SkillNotCachedError",
    "is_registry_ref",
    "parse_registry_ref",
    "resolve_registry_ref",
]
