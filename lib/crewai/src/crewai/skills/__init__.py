"""Agent Skills standard implementation for crewAI.

Provides filesystem-based skill packaging with progressive disclosure.
"""

from crewai.skills.loader import activate_skill, discover_skills
from crewai.skills.models import Skill, SkillFrontmatter
from crewai.skills.parser import SkillParseError


__all__ = [
    "Skill",
    "SkillFrontmatter",
    "SkillParseError",
    "activate_skill",
    "discover_skills",
]
