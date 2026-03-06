"""Pydantic data models for the Agent Skills standard.

Defines DisclosureLevel, SkillFrontmatter, and Skill models for
progressive disclosure of skill information.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from crewai.skills.validation import validate_skill_name


class DisclosureLevel(IntEnum):
    """Progressive disclosure levels for skill loading.

    Attributes:
        METADATA: Only frontmatter metadata is loaded (name, description).
        INSTRUCTIONS: Full SKILL.md body is loaded.
        RESOURCES: Resource directories (scripts, references, assets) are cataloged.
    """

    METADATA = 1
    INSTRUCTIONS = 2
    RESOURCES = 3


class SkillFrontmatter(BaseModel, frozen=True):
    """YAML frontmatter from a SKILL.md file.

    Attributes:
        name: Unique skill identifier (1-64 chars, lowercase alphanumeric + hyphens).
        description: Human-readable description of the skill.
        license: Optional SPDX license identifier.
        compatibility: Optional compatibility information.
        metadata: Optional additional metadata as string key-value pairs.
        allowed_tools: Optional list of tools the skill may use.
    """

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: list[str] | None = None

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str) -> str:
        """Validate skill name against spec constraints."""
        return validate_skill_name(v)


class Skill(BaseModel):
    """A loaded Agent Skill with progressive disclosure support.

    Attributes:
        frontmatter: Parsed YAML frontmatter.
        instructions: Full SKILL.md body text (populated at INSTRUCTIONS level).
        path: Filesystem path to the skill directory.
        disclosure_level: Current disclosure level of the skill.
        resource_files: Cataloged resource files (populated at RESOURCES level).
    """

    frontmatter: SkillFrontmatter
    instructions: str | None = None
    path: Path
    disclosure_level: DisclosureLevel = Field(default=DisclosureLevel.METADATA)
    resource_files: dict[str, list[str]] | None = None

    @property
    def name(self) -> str:
        """Skill name from frontmatter."""
        return self.frontmatter.name

    @property
    def description(self) -> str:
        """Skill description from frontmatter."""
        return self.frontmatter.description

    @property
    def scripts_dir(self) -> Path:
        """Path to the scripts directory."""
        return self.path / "scripts"

    @property
    def references_dir(self) -> Path:
        """Path to the references directory."""
        return self.path / "references"

    @property
    def assets_dir(self) -> Path:
        """Path to the assets directory."""
        return self.path / "assets"

    def has_scripts(self) -> bool:
        """Check if the skill has a scripts directory."""
        return self.scripts_dir.is_dir()

    def has_references(self) -> bool:
        """Check if the skill has a references directory."""
        return self.references_dir.is_dir()

    def has_assets(self) -> bool:
        """Check if the skill has an assets directory."""
        return self.assets_dir.is_dir()

    def with_disclosure_level(
        self,
        level: DisclosureLevel,
        instructions: str | None = None,
        resource_files: dict[str, list[str]] | None = None,
    ) -> Skill:
        """Create a new Skill at a different disclosure level.

        Args:
            level: The new disclosure level.
            instructions: Optional instructions body text.
            resource_files: Optional cataloged resource files.

        Returns:
            A new Skill instance at the specified disclosure level.
        """
        return Skill(
            frontmatter=self.frontmatter,
            instructions=instructions
            if instructions is not None
            else self.instructions,
            path=self.path,
            disclosure_level=level,
            resource_files=(
                resource_files if resource_files is not None else self.resource_files
            ),
        )
