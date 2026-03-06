"""SKILL.md file parsing for the Agent Skills standard.

Parses YAML frontmatter and markdown body from SKILL.md files,
and provides progressive loading functions for skill data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from crewai.skills.models import DisclosureLevel, Skill, SkillFrontmatter
from crewai.skills.validation import validate_directory_name


SKILL_FILENAME: str = "SKILL.md"


class SkillParseError(ValueError):
    """Error raised when SKILL.md parsing fails."""


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md content into frontmatter dict and body text.

    Args:
        content: Raw SKILL.md file content.

    Returns:
        Tuple of (frontmatter dict, body text).

    Raises:
        SkillParseError: If frontmatter delimiters are missing or YAML is invalid.
    """
    if not content.startswith("---"):
        msg = "SKILL.md must start with '---' frontmatter delimiter"
        raise SkillParseError(msg)

    end_idx = content.find("---", 3)
    if end_idx == -1:
        msg = "SKILL.md missing closing '---' frontmatter delimiter"
        raise SkillParseError(msg)

    yaml_content = content[3:end_idx].strip()
    body = content[end_idx + 3 :].strip()

    try:
        frontmatter = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML in frontmatter: {e}"
        raise SkillParseError(msg) from e

    if not isinstance(frontmatter, dict):
        msg = "Frontmatter must be a YAML mapping"
        raise SkillParseError(msg)

    return frontmatter, body


def parse_skill_md(path: Path) -> tuple[SkillFrontmatter, str]:
    """Read and parse a SKILL.md file.

    Args:
        path: Path to the SKILL.md file.

    Returns:
        Tuple of (SkillFrontmatter, body text).

    Raises:
        FileNotFoundError: If the file does not exist.
        SkillParseError: If parsing fails.
    """
    content = path.read_text(encoding="utf-8")
    frontmatter_dict, body = parse_frontmatter(content)
    frontmatter = SkillFrontmatter(**frontmatter_dict)
    return frontmatter, body


def load_skill_metadata(skill_dir: Path) -> Skill:
    """Load a skill at METADATA disclosure level.

    Parses SKILL.md frontmatter only and validates directory name.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        Skill instance at METADATA level.

    Raises:
        FileNotFoundError: If SKILL.md is missing.
        SkillParseError: If parsing fails.
        ValueError: If directory name doesn't match skill name.
    """
    skill_md_path = skill_dir / SKILL_FILENAME
    frontmatter, _body = parse_skill_md(skill_md_path)
    validate_directory_name(skill_dir, frontmatter.name)
    return Skill(
        frontmatter=frontmatter,
        path=skill_dir,
        disclosure_level=DisclosureLevel.METADATA,
    )


def load_skill_instructions(skill: Skill) -> Skill:
    """Promote a skill to INSTRUCTIONS disclosure level.

    Reads the full SKILL.md body text.

    Args:
        skill: Skill at METADATA level.

    Returns:
        New Skill instance at INSTRUCTIONS level.
    """
    if skill.disclosure_level >= DisclosureLevel.INSTRUCTIONS:
        return skill

    skill_md_path = skill.path / SKILL_FILENAME
    _, body = parse_skill_md(skill_md_path)
    return skill.with_disclosure_level(
        level=DisclosureLevel.INSTRUCTIONS,
        instructions=body,
    )


def load_skill_resources(skill: Skill) -> Skill:
    """Promote a skill to RESOURCES disclosure level.

    Catalogs available resource directories (scripts, references, assets).

    Args:
        skill: Skill at any level.

    Returns:
        New Skill instance at RESOURCES level.
    """
    if skill.disclosure_level >= DisclosureLevel.RESOURCES:
        return skill

    if skill.disclosure_level < DisclosureLevel.INSTRUCTIONS:
        skill = load_skill_instructions(skill)

    resource_files: dict[str, list[str]] = {}
    for dir_name in ("scripts", "references", "assets"):
        resource_dir = skill.path / dir_name
        if resource_dir.is_dir():
            resource_files[dir_name] = sorted(
                str(f.relative_to(resource_dir))
                for f in resource_dir.rglob("*")
                if f.is_file()
            )

    return skill.with_disclosure_level(
        level=DisclosureLevel.RESOURCES,
        instructions=skill.instructions,
        resource_files=resource_files,
    )
