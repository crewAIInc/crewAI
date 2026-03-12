"""Tests for skills/models.py."""

from pathlib import Path

import pytest

from crewai.skills.models import (
    INSTRUCTIONS,
    METADATA,
    RESOURCES,
    Skill,
    SkillFrontmatter,
)


class TestDisclosureLevel:
    """Tests for DisclosureLevel constants."""

    def test_ordering(self) -> None:
        assert METADATA < INSTRUCTIONS
        assert INSTRUCTIONS < RESOURCES

    def test_values(self) -> None:
        assert METADATA == 1
        assert INSTRUCTIONS == 2
        assert RESOURCES == 3


class TestSkillFrontmatter:
    """Tests for SkillFrontmatter model."""

    def test_required_fields(self) -> None:
        fm = SkillFrontmatter(name="my-skill", description="A test skill")
        assert fm.name == "my-skill"
        assert fm.description == "A test skill"
        assert fm.license is None
        assert fm.metadata is None
        assert fm.allowed_tools is None

    def test_all_fields(self) -> None:
        fm = SkillFrontmatter(
            name="web-search",
            description="Search the web",
            license="Apache-2.0",
            compatibility="crewai>=0.1.0",
            metadata={"author": "test"},
            allowed_tools=["browser"],
        )
        assert fm.license == "Apache-2.0"
        assert fm.metadata == {"author": "test"}
        assert fm.allowed_tools == ["browser"]

    def test_frozen(self) -> None:
        fm = SkillFrontmatter(name="my-skill", description="desc")
        with pytest.raises(Exception):
            fm.name = "other"  # type: ignore[misc]

    def test_invalid_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            SkillFrontmatter(name="Invalid--Name", description="bad")


class TestSkill:
    """Tests for Skill model."""

    def test_properties(self, tmp_path: Path) -> None:
        fm = SkillFrontmatter(name="test-skill", description="desc")
        skill = Skill(frontmatter=fm, path=tmp_path / "test-skill")
        assert skill.name == "test-skill"
        assert skill.description == "desc"
        assert skill.disclosure_level == METADATA

    def test_resource_dirs(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        fm = SkillFrontmatter(name="test-skill", description="desc")
        skill = Skill(frontmatter=fm, path=skill_dir)
        assert skill.scripts_dir == skill_dir / "scripts"
        assert skill.references_dir == skill_dir / "references"
        assert skill.assets_dir == skill_dir / "assets"

    def test_with_disclosure_level(self, tmp_path: Path) -> None:
        fm = SkillFrontmatter(name="test-skill", description="desc")
        skill = Skill(frontmatter=fm, path=tmp_path)
        promoted = skill.with_disclosure_level(
            INSTRUCTIONS,
            instructions="Do this.",
        )
        assert promoted.disclosure_level == INSTRUCTIONS
        assert promoted.instructions == "Do this."
        assert skill.disclosure_level == METADATA
