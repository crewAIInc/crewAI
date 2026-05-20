"""Agent Skills standard implementation for crewAI.

Provides filesystem-based skill packaging with progressive disclosure.
"""

from crewai.skills.cache import SkillCacheManager
from crewai.skills.loader import activate_skill, discover_skills
from crewai.skills.models import Skill, SkillFrontmatter
from crewai.skills.parser import SkillParseError
from crewai.skills.registry import is_registry_ref, resolve_registry_ref


__all__ = [
    "Skill",
    "SkillCacheManager",
    "SkillFrontmatter",
    "SkillParseError",
    "activate_skill",
    "discover_skills",
    "is_registry_ref",
    "resolve_registry_ref",
]
