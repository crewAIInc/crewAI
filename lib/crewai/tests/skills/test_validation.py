"""Tests for skills/validation.py."""

from pathlib import Path

import pytest

from crewai.skills.validation import (
    MAX_SKILL_NAME_LENGTH,
    validate_directory_name,
    validate_skill_name,
)


class TestValidateSkillName:
    """Tests for validate_skill_name."""

    def test_simple_name(self) -> None:
        assert validate_skill_name("web-search") == "web-search"

    def test_single_word(self) -> None:
        assert validate_skill_name("search") == "search"

    def test_numeric(self) -> None:
        assert validate_skill_name("tool3") == "tool3"

    def test_all_digits(self) -> None:
        assert validate_skill_name("123") == "123"

    def test_single_char(self) -> None:
        assert validate_skill_name("a") == "a"

    def test_max_length(self) -> None:
        name = "a" * MAX_SKILL_NAME_LENGTH
        assert validate_skill_name(name) == name

    def test_multi_hyphen_segments(self) -> None:
        assert validate_skill_name("my-cool-skill") == "my-cool-skill"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            validate_skill_name("")

    def test_too_long_raises(self) -> None:
        name = "a" * (MAX_SKILL_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="at most"):
            validate_skill_name(name)

    def test_uppercase_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("MySkill")

    def test_leading_hyphen_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("-skill")

    def test_trailing_hyphen_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("skill-")

    def test_consecutive_hyphens_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("my--skill")

    def test_underscore_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("my_skill")

    def test_space_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("my skill")

    def test_special_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("skill@v1")


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
