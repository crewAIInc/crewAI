"""Validation functions for Agent Skills specification constraints.

Validates skill names and directory structures per the Agent Skills standard.
"""

from __future__ import annotations

from pathlib import Path
import re


MAX_SKILL_NAME_LENGTH: int = 64
MIN_SKILL_NAME_LENGTH: int = 1

_SKILL_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_skill_name(name: str) -> str:
    """Validate a skill name against the Agent Skills specification.

    Names must be 1-64 characters, lowercase alphanumeric with hyphens,
    no leading/trailing hyphens, and no consecutive hyphens.

    Args:
        name: The skill name to validate.

    Returns:
        The validated skill name.

    Raises:
        ValueError: If the name violates any constraint.
    """
    if len(name) < MIN_SKILL_NAME_LENGTH:
        msg = "Skill name must not be empty"
        raise ValueError(msg)

    if len(name) > MAX_SKILL_NAME_LENGTH:
        msg = (
            f"Skill name must be at most {MAX_SKILL_NAME_LENGTH} characters, "
            f"got {len(name)}"
        )
        raise ValueError(msg)

    if not _SKILL_NAME_PATTERN.match(name):
        msg = (
            f"Invalid skill name '{name}'. Names must be lowercase alphanumeric "
            f"with single hyphens, no leading/trailing hyphens."
        )
        raise ValueError(msg)

    return name


def validate_directory_name(skill_dir: Path, skill_name: str) -> None:
    """Validate that a directory name matches the skill name.

    Args:
        skill_dir: Path to the skill directory.
        skill_name: The declared skill name from frontmatter.

    Raises:
        ValueError: If the directory name does not match the skill name.
    """
    dir_name = skill_dir.name
    if dir_name != skill_name:
        msg = f"Directory name '{dir_name}' does not match skill name '{skill_name}'"
        raise ValueError(msg)
