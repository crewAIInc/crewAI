"""Tests for skills validation."""

from pathlib import Path

import pytest

from crewai.skills.models import SkillFrontmatter
from crewai.skills.validation import (
    MAX_SKILL_NAME_LENGTH,
    validate_directory_name,
)


def _make(name: str) -> SkillFrontmatter:
    """Create a SkillFrontmatter with the given name."""
    return SkillFrontmatter(name=name, description="desc")


class TestSkillNameValidation:
    """Tests for skill name constraints via SkillFrontmatter."""

    def test_simple_name(self) -> None:
        assert _make("web-search").name == "web-search"

    def test_single_word(self) -> None:
        assert _make("search").name == "search"

    def test_numeric(self) -> None:
        assert _make("tool3").name == "tool3"

    def test_all_digits(self) -> None:
        assert _make("123").name == "123"

    def test_single_char(self) -> None:
        assert _make("a").name == "a"

    def test_max_length(self) -> None:
        name = "a" * MAX_SKILL_NAME_LENGTH
        assert _make(name).name == name

    def test_multi_hyphen_segments(self) -> None:
        assert _make("my-cool-skill").name == "my-cool-skill"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("")

    def test_too_long_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("a" * (MAX_SKILL_NAME_LENGTH + 1))

    def test_uppercase_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("MySkill")

    def test_leading_hyphen_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("-skill")

    def test_trailing_hyphen_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("skill-")

    def test_consecutive_hyphens_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("my--skill")

    def test_underscore_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("my_skill")

    def test_space_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("my skill")

    def test_special_chars_raises(self) -> None:
        with pytest.raises(ValueError):
            _make("skill@v1")


class TestValidateDirectoryName:
    """Tests for validate_directory_name."""

    def test_matching_names(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        validate_directory_name(skill_dir, "my-skill")

    def test_mismatched_names(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "other-name"
        skill_dir.mkdir()
        with pytest.raises(ValueError, match="does not match"):
            validate_directory_name(skill_dir, "my-skill")
