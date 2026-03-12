"""Tests for skills/loader.py."""

from pathlib import Path

import pytest

from crewai.skills.loader import (
    activate_skill,
    discover_skills,
    format_skill_context,
    load_resources,
)
from crewai.skills.models import INSTRUCTIONS, METADATA, RESOURCES, Skill, SkillFrontmatter
from crewai.skills.parser import load_skill_metadata


def _create_skill_dir(parent: Path, name: str, body: str = "Body.") -> Path:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = parent / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Skill {name}\n---\n{body}"
    )
    return skill_dir


class TestDiscoverSkills:
    """Tests for discover_skills."""

    def test_finds_valid_skills(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "alpha")
        _create_skill_dir(tmp_path, "beta")
        skills = discover_skills(tmp_path)
        names = {s.name for s in skills}
        assert names == {"alpha", "beta"}

    def test_skips_dirs_without_skill_md(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "valid")
        (tmp_path / "no-skill").mkdir()
        skills = discover_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "valid"

    def test_skips_invalid_skills(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "good-skill")
        bad_dir = tmp_path / "bad-skill"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text(
            "---\nname: Wrong-Name\ndescription: bad\n---\n"
        )
        skills = discover_skills(tmp_path)
        assert len(skills) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        skills = discover_skills(tmp_path)
        assert skills == []

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_skills(tmp_path / "nonexistent")

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "zebra")
        _create_skill_dir(tmp_path, "alpha")
        skills = discover_skills(tmp_path)
        assert [s.name for s in skills] == ["alpha", "zebra"]


class TestActivateSkill:
    """Tests for activate_skill."""

    def test_promotes_to_instructions(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "my-skill", body="Instructions.")
        skill = load_skill_metadata(tmp_path / "my-skill")
        activated = activate_skill(skill)
        assert activated.disclosure_level == INSTRUCTIONS
        assert activated.instructions == "Instructions."

    def test_idempotent(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "my-skill")
        skill = load_skill_metadata(tmp_path / "my-skill")
        activated = activate_skill(skill)
        again = activate_skill(activated)
        assert again is activated


class TestLoadResources:
    """Tests for load_resources."""

    def test_promotes_to_resources(self, tmp_path: Path) -> None:
        skill_dir = _create_skill_dir(tmp_path, "my-skill")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "run.sh").write_text("#!/bin/bash")
        skill = load_skill_metadata(skill_dir)
        full = load_resources(skill)
        assert full.disclosure_level == RESOURCES


class TestFormatSkillContext:
    """Tests for format_skill_context."""

    def test_metadata_level(self, tmp_path: Path) -> None:
        fm = SkillFrontmatter(name="test-skill", description="A skill")
        skill = Skill(
            frontmatter=fm, path=tmp_path, disclosure_level=METADATA
        )
        ctx = format_skill_context(skill)
        assert "## Skill: test-skill" in ctx
        assert "A skill" in ctx

    def test_instructions_level(self, tmp_path: Path) -> None:
        fm = SkillFrontmatter(name="test-skill", description="A skill")
        skill = Skill(
            frontmatter=fm,
            path=tmp_path,
            disclosure_level=INSTRUCTIONS,
            instructions="Do these things.",
        )
        ctx = format_skill_context(skill)
        assert "## Skill: test-skill" in ctx
        assert "Do these things." in ctx

    def test_no_instructions_at_instructions_level(self, tmp_path: Path) -> None:
        fm = SkillFrontmatter(name="test-skill", description="A skill")
        skill = Skill(
            frontmatter=fm,
            path=tmp_path,
            disclosure_level=INSTRUCTIONS,
            instructions=None,
        )
        ctx = format_skill_context(skill)
        assert ctx == "## Skill: test-skill\nA skill"

    def test_resources_level(self, tmp_path: Path) -> None:
        fm = SkillFrontmatter(name="test-skill", description="A skill")
        skill = Skill(
            frontmatter=fm,
            path=tmp_path,
            disclosure_level=RESOURCES,
            instructions="Do things.",
            resource_files={
                "scripts": ["run.sh"],
                "assets": ["data.json", "config.yaml"],
            },
        )
        ctx = format_skill_context(skill)
        assert "### Available Resources" in ctx
        assert "**assets/**: data.json, config.yaml" in ctx
        assert "**scripts/**: run.sh" in ctx

    def test_resources_level_empty_files(self, tmp_path: Path) -> None:
        fm = SkillFrontmatter(name="test-skill", description="A skill")
        skill = Skill(
            frontmatter=fm,
            path=tmp_path,
            disclosure_level=RESOURCES,
            instructions="Do things.",
            resource_files={},
        )
        ctx = format_skill_context(skill)
        assert "### Available Resources" not in ctx
