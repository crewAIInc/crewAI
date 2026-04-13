"""Tests for skills/parser.py."""

from pathlib import Path

import pytest

from crewai.skills.models import INSTRUCTIONS, METADATA, RESOURCES
from crewai.skills.parser import (
    SkillParseError,
    load_skill_instructions,
    load_skill_metadata,
    load_skill_resources,
    parse_frontmatter,
    parse_skill_md,
)


class TestParseFrontmatter:
    """Tests for parse_frontmatter."""

    def test_valid_frontmatter_and_body(self) -> None:
        content = "---\nname: test\ndescription: A test\n---\n\nBody text here."
        fm, body = parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "A test"
        assert body == "Body text here."

    def test_empty_body(self) -> None:
        content = "---\nname: test\ndescription: A test\n---"
        fm, body = parse_frontmatter(content)
        assert fm["name"] == "test"
        assert body == ""

    def test_missing_opening_delimiter(self) -> None:
        with pytest.raises(SkillParseError, match="must start with"):
            parse_frontmatter("name: test\n---\nBody")

    def test_missing_closing_delimiter(self) -> None:
        with pytest.raises(SkillParseError, match="missing closing"):
            parse_frontmatter("---\nname: test\n")

    def test_invalid_yaml(self) -> None:
        with pytest.raises(SkillParseError, match="Invalid YAML"):
            parse_frontmatter("---\n: :\n  bad: [yaml\n---\nBody")

    def test_triple_dash_in_body(self) -> None:
        content = "---\nname: test\ndescription: desc\n---\n\nBody with --- inside."
        fm, body = parse_frontmatter(content)
        assert "---" in body

    def test_inline_triple_dash_in_yaml_value(self) -> None:
        content = '---\nname: test\ndescription: "Use---carefully"\n---\n\nBody.'
        fm, body = parse_frontmatter(content)
        assert fm["description"] == "Use---carefully"
        assert body == "Body."

    def test_unicode_content(self) -> None:
        content = "---\nname: test\ndescription: Beschreibung\n---\n\nUnicode: \u00e4\u00f6\u00fc\u00df"
        fm, body = parse_frontmatter(content)
        assert fm["description"] == "Beschreibung"
        assert "\u00e4\u00f6\u00fc\u00df" in body

    def test_non_mapping_frontmatter(self) -> None:
        with pytest.raises(SkillParseError, match="must be a YAML mapping"):
            parse_frontmatter("---\n- item1\n- item2\n---\nBody")


class TestParseSkillMd:
    """Tests for parse_skill_md."""

    def test_valid_file(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\nname: my-skill\ndescription: desc\n---\nInstructions here."
        )
        fm, body = parse_skill_md(skill_md)
        assert fm.name == "my-skill"
        assert body == "Instructions here."

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_skill_md(tmp_path / "nonexistent" / "SKILL.md")


class TestLoadSkillMetadata:
    """Tests for load_skill_metadata."""

    def test_valid_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test skill\n---\nBody"
        )
        skill = load_skill_metadata(skill_dir)
        assert skill.name == "my-skill"
        assert skill.disclosure_level == METADATA
        assert skill.instructions is None

    def test_directory_name_mismatch(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "wrong-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test skill\n---\n"
        )
        with pytest.raises(ValueError, match="does not match"):
            load_skill_metadata(skill_dir)


class TestLoadSkillInstructions:
    """Tests for load_skill_instructions."""

    def test_promotes_to_instructions(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test\n---\nFull body."
        )
        skill = load_skill_metadata(skill_dir)
        promoted = load_skill_instructions(skill)
        assert promoted.disclosure_level == INSTRUCTIONS
        assert promoted.instructions == "Full body."

    def test_idempotent(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test\n---\nBody."
        )
        skill = load_skill_metadata(skill_dir)
        promoted = load_skill_instructions(skill)
        again = load_skill_instructions(promoted)
        assert again is promoted


class TestLoadSkillResources:
    """Tests for load_skill_resources."""

    def test_catalogs_resources(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test\n---\nBody."
        )
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "run.sh").write_text("#!/bin/bash")
        (skill_dir / "assets").mkdir()
        (skill_dir / "assets" / "data.json").write_text("{}")

        skill = load_skill_metadata(skill_dir)
        full = load_skill_resources(skill)
        assert full.disclosure_level == RESOURCES
        assert full.instructions == "Body."
        assert full.resource_files is not None
        assert "scripts" in full.resource_files
        assert "run.sh" in full.resource_files["scripts"]
        assert "assets" in full.resource_files
        assert "data.json" in full.resource_files["assets"]

    def test_no_resource_dirs(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test\n---\nBody."
        )
        skill = load_skill_metadata(skill_dir)
        full = load_skill_resources(skill)
        assert full.resource_files == {}
