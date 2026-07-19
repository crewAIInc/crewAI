"""Deprecated location for the Skills Repository.

The registry-backed skills feature graduated out of experimental; it now
lives in `crewai.skills` alongside the filesystem-based loader. This module
re-exports the public names and aliases the old submodules so imports
written against the experimental namespace keep working, and will be
removed in a future release.
"""

import sys

from crewai.skills import cache, events, registry
from crewai.skills.cache import SkillCacheManager
from crewai.skills.registry import (
    SkillNotCachedError,
    is_registry_ref,
    parse_registry_ref,
    resolve_registry_ref,
)


# Keep `crewai.experimental.skills.<module>` imports (and patch targets)
# resolving to the real modules in crewai.skills.
sys.modules[__name__ + ".cache"] = cache
sys.modules[__name__ + ".events"] = events
sys.modules[__name__ + ".registry"] = registry

__all__ = [
    "SkillCacheManager",
    "SkillNotCachedError",
    "cache",
    "events",
    "is_registry_ref",
    "parse_registry_ref",
    "registry",
    "resolve_registry_ref",
]
