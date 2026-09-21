"""Validation functions for Agent Skills specification constraints.

Validates skill names and directory structures per the Agent Skills standard.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final


MAX_SKILL_NAME_LENGTH: Final[int] = 64
MIN_SKILL_NAME_LENGTH: Final[int] = 1
SKILL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def versions_match(left: Any, right: Any) -> bool:
    """Return True when two skill versions refer to the same version.

    Versions arrive as ``1.2.0`` or ``v1.2.0`` depending on whether they came
    from a registry ref, SKILL.md frontmatter or a cache entry, so comparison
    ignores case and an optional leading ``v``.

    Anything that isn't a version string — None, or a corrupted cache entry
    holding a number — never matches: a pinned reference should re-resolve
    rather than accept a version it can't confirm.
    """
    normalized_left = _normalize_version(left)
    normalized_right = _normalize_version(right)
    if normalized_left is None or normalized_right is None:
        return False
    return normalized_left == normalized_right


def _normalize_version(value: Any) -> str | None:
    """Return a comparable form of a version, or None when it isn't one."""
    if not isinstance(value, str):
        return None
    return value.strip().lower().removeprefix("v") or None


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
