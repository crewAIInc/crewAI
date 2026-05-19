"""Tests for the version field added to SkillFrontmatter."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crewai.skills.models import SkillFrontmatter


class TestSkillFrontmatterVersion:
    def test_version_defaults_to_none(self) -> None:
        fm = SkillFrontmatter(name="my-skill", description="A skill.")
        assert fm.version is None

    def test_version_can_be_set(self) -> None:
        fm = SkillFrontmatter(name="my-skill", description="A skill.", version="1.2.3")
        assert fm.version == "1.2.3"

    def test_existing_frontmatter_without_version_still_valid(self) -> None:
        """Backward compat: existing SKILL.md files without version must still parse."""
        fm = SkillFrontmatter(name="old-skill", description="Old skill without version.")
        assert fm.version is None

    def test_version_is_optional_string(self) -> None:
        fm = SkillFrontmatter(name="my-skill", description="Desc.", version=None)
        assert fm.version is None

    def test_frontmatter_is_frozen(self) -> None:
        fm = SkillFrontmatter(name="my-skill", description="A skill.", version="1.0.0")
        with pytest.raises(ValidationError):
            fm.version = "2.0.0"  # type: ignore[misc]
