"""Tests for skills/loader.py."""

from pathlib import Path

import pytest

from crewai.skills.loader import (
    activate_skill,
    discover_skills,
    format_skill_context,
    load_resources,
    load_skill,
    load_skills,
)
from crewai.skills.models import INSTRUCTIONS, METADATA, RESOURCES, Skill, SkillFrontmatter
from crewai.skills.parser import SkillParseError, load_skill_metadata


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


class TestLoadSkill:
    """Tests for load_skill."""

    @pytest.mark.parametrize("as_string", [False, True])
    def test_loads_path_input(self, tmp_path: Path, as_string: bool) -> None:
        _create_skill_dir(tmp_path, "first-skill", body="First.")
        _create_skill_dir(tmp_path, "second-skill", body="Second.")
        path = str(tmp_path) if as_string else tmp_path

        skills = load_skill(path)

        assert [skill.name for skill in skills] == ["first-skill", "second-skill"]
        assert [skill.disclosure_level for skill in skills] == [
            INSTRUCTIONS,
            INSTRUCTIONS,
        ]
        assert [skill.instructions for skill in skills] == ["First.", "Second."]

    def test_loads_preloaded_skill(self, tmp_path: Path) -> None:
        preloaded = Skill(
            frontmatter=SkillFrontmatter(
                name="preloaded-skill",
                description="Preloaded skill",
            ),
            path=tmp_path / "preloaded-skill",
        )

        skills = load_skill(preloaded)

        assert skills == [preloaded]

    def test_loads_inline_skill(self) -> None:
        inline_skill = (
            "---\n"
            "name: inline-skill\n"
            "description: Inline guidance\n"
            "---\n"
            "Follow these instructions."
        )

        skills = load_skill(inline_skill)

        assert [skill.name for skill in skills] == ["inline-skill"]
        assert [skill.disclosure_level for skill in skills] == [INSTRUCTIONS]
        assert [skill.instructions for skill in skills] == [
            "Follow these instructions."
        ]

    def test_invalid_inline_skill_raises_parse_error(self) -> None:
        with pytest.raises(SkillParseError, match="missing closing"):
            load_skill("---\nname: inline-skill\n")

    def test_missing_path_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_skill(tmp_path / "missing")

    def test_unsupported_input_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Unsupported skill input"):
            load_skill(object())  # type: ignore[arg-type]

    def test_load_skills_deduplicates_by_name(self, tmp_path: Path) -> None:
        first = Skill(
            frontmatter=SkillFrontmatter(
                name="duplicate-skill",
                description="First skill",
            ),
            path=tmp_path / "first",
        )
        second = Skill(
            frontmatter=SkillFrontmatter(
                name="duplicate-skill",
                description="Second skill",
            ),
            path=tmp_path / "second",
        )

        skills = load_skills([first, second])

        assert skills == [first]

    def test_load_skills_keeps_registry_refs_from_different_orgs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        first = Skill(
            frontmatter=SkillFrontmatter(
                name="shared-skill",
                description="First registry skill",
            ),
            path=tmp_path / "first",
            disclosure_level=INSTRUCTIONS,
            instructions="First instructions.",
        )
        second = Skill(
            frontmatter=SkillFrontmatter(
                name="shared-skill",
                description="Second registry skill",
            ),
            path=tmp_path / "second",
            disclosure_level=INSTRUCTIONS,
            instructions="Second instructions.",
        )

        def resolve_registry_ref(ref: str, source: object = None) -> Skill:
            return {
                "@first/shared-skill": first,
                "@second/shared-skill": second,
            }[ref]

        monkeypatch.setattr(
            "crewai.skills.registry.resolve_registry_ref",
            resolve_registry_ref,
        )

        skills = load_skills(["@first/shared-skill", "@second/shared-skill"])

        assert skills == [first, second]


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
        assert '<skill name="test-skill">' in ctx
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
        assert '<skill name="test-skill">' in ctx
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
        assert ctx == '<skill name="test-skill">\nA skill\n</skill>'

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
