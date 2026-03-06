"""SKILL.md file parsing for the Agent Skills standard.

Parses YAML frontmatter and markdown body from SKILL.md files,
and provides progressive loading functions for skill data.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from crewai.skills.models import DisclosureLevel, Skill, SkillFrontmatter
from crewai.skills.validation import validate_directory_name


SKILL_FILENAME: str = "SKILL.md"
_CLOSING_DELIMITER: re.Pattern[str] = re.compile(r"\n---[ \t]*(?:\n|$)")


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

    match = _CLOSING_DELIMITER.search(content, pos=3)
    if match is None:
        msg = "SKILL.md missing closing '---' frontmatter delimiter"
        raise SkillParseError(msg)

    yaml_content = content[3 : match.start()].strip()
    body = content[match.end() :].strip()

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
    for dir_name, resource_dir in (
        ("scripts", skill.scripts_dir),
        ("references", skill.references_dir),
        ("assets", skill.assets_dir),
    ):
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
