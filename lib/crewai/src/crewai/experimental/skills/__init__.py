"""Deprecated location for the Skills Repository.

The registry-backed skills feature graduated out of experimental; it now
lives in `crewai.skills` alongside the filesystem-based loader. This module
re-exports the public names so imports written against the experimental
namespace keep working, and will be removed in a future release.
"""

from crewai.skills.cache import SkillCacheManager
from crewai.skills.registry import (
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
