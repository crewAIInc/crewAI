"""Agent Skills standard implementation for crewAI.

Provides filesystem-based skill packaging with progressive disclosure, plus
the registry-backed Skills Repository (`@org/name` refs, global cache,
downloads).
"""

from crewai.skills.cache import SkillCacheManager
from crewai.skills.loader import (
    activate_skill,
    discover_skills,
    load_skill,
    load_skills,
)
from crewai.skills.models import Skill, SkillFrontmatter
from crewai.skills.parser import SkillParseError
from crewai.skills.registry import (
    SkillNotCachedError,
    is_registry_ref,
    parse_registry_ref,
    resolve_registry_ref,
)


__all__ = [
    "Skill",
    "SkillCacheManager",
    "SkillFrontmatter",
    "SkillNotCachedError",
    "SkillParseError",
    "activate_skill",
    "discover_skills",
    "is_registry_ref",
    "load_skill",
    "load_skills",
    "parse_registry_ref",
    "resolve_registry_ref",
]
