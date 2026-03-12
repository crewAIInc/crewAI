"""Validation functions for Agent Skills specification constraints.

Validates skill names and directory structures per the Agent Skills standard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


MAX_SKILL_NAME_LENGTH: int = 64
MIN_SKILL_NAME_LENGTH: int = 1
SKILL_NAME_PATTERN: str = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


def coerce_skill_paths(v: list[Any] | None) -> list[Any] | None:
    """Coerce string entries to Path objects, pass through other types.

    Args:
        v: List of skill paths or Skill objects, or None.

    Returns:
        The list with string entries converted to Path objects, or None.
    """
    if not v:
        return v
    return [Path(item) if isinstance(item, str) else item for item in v]


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
