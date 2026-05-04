"""Pydantic data models for the Agent Skills standard.

Defines DisclosureLevel, SkillFrontmatter, and Skill models for
progressive disclosure of skill information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crewai.skills.validation import (
    MAX_SKILL_NAME_LENGTH,
    MIN_SKILL_NAME_LENGTH,
    SKILL_NAME_PATTERN,
)


MAX_DESCRIPTION_LENGTH: Final[int] = 1024
ResourceDirName = Literal["scripts", "references", "assets"]


DisclosureLevel = Annotated[
    Literal[1, 2, 3], "Progressive disclosure levels for skill loading."
]

METADATA: Final[
    Annotated[
        DisclosureLevel, "Only frontmatter metadata is loaded (name, description)."
    ]
] = 1
INSTRUCTIONS: Final[Annotated[DisclosureLevel, "Full SKILL.md body is loaded."]] = 2
RESOURCES: Final[
    Annotated[
        DisclosureLevel,
        "Resource directories (scripts, references, assets) are cataloged.",
    ]
] = 3


class SkillFrontmatter(BaseModel):
    """YAML frontmatter from a SKILL.md file.

    Attributes:
        name: Unique skill identifier (1-64 chars, lowercase alphanumeric + hyphens).
        description: Human-readable description (1-1024 chars).
        license: Optional license name or reference.
        compatibility: Optional compatibility information (max 500 chars).
        metadata: Optional additional metadata as string key-value pairs.
        allowed_tools: Optional space-delimited list of pre-approved tools.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str = Field(
        min_length=MIN_SKILL_NAME_LENGTH,
        max_length=MAX_SKILL_NAME_LENGTH,
        pattern=SKILL_NAME_PATTERN,
    )
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_LENGTH)
    license: str | None = Field(
        default=None,
        description="SPDX license identifier or free-text license reference, e.g. 'MIT', 'Apache-2.0'.",
    )
    compatibility: str | None = Field(
        default=None,
        max_length=500,
        description="Version or platform constraints for the skill, e.g. 'crewai >= 0.80'.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Arbitrary string key-value pairs for custom skill metadata.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        alias="allowed-tools",
        description="Pre-approved tool names the skill may use, parsed from a space-delimited string in frontmatter.",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_allowed_tools(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Parse space-delimited allowed-tools string into a list."""
        key = "allowed-tools"
        alt_key = "allowed_tools"
        raw = values.get(key) or values.get(alt_key)
        if isinstance(raw, str):
            values[key] = raw.split()
        return values


class Skill(BaseModel):
    """A loaded Agent Skill with progressive disclosure support.

    Attributes:
        frontmatter: Parsed YAML frontmatter.
        instructions: Full SKILL.md body text (populated at INSTRUCTIONS level).
        path: Filesystem path to the skill directory.
        disclosure_level: Current disclosure level of the skill.
        resource_files: Cataloged resource files (populated at RESOURCES level).
    """

    frontmatter: SkillFrontmatter = Field(
        description="Parsed YAML frontmatter from SKILL.md.",
    )
    instructions: str | None = Field(
        default=None,
        description="Full SKILL.md body text, populated at INSTRUCTIONS level.",
    )
    path: Path = Field(
        description="Filesystem path to the skill directory.",
    )
    disclosure_level: DisclosureLevel = Field(
        default=METADATA,
        description="Current progressive disclosure level of the skill.",
    )
    resource_files: dict[ResourceDirName, list[str]] | None = Field(
        default=None,
        description="Cataloged resource files by directory, populated at RESOURCES level.",
    )

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

    def with_disclosure_level(
        self,
        level: DisclosureLevel,
        instructions: str | None = None,
        resource_files: dict[ResourceDirName, list[str]] | None = None,
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
