"""Tests for the 'version' metadata key on SkillFrontmatter."""

from __future__ import annotations

from crewai.skills.models import SkillFrontmatter


class TestSkillFrontmatterVersion:
    def test_version_defaults_to_no_metadata(self) -> None:
        fm = SkillFrontmatter(name="my-skill", description="A skill.")
        assert fm.metadata is None

    def test_version_via_metadata(self) -> None:
        fm = SkillFrontmatter(
            name="my-skill",
            description="A skill.",
            metadata={"version": "1.2.3"},
        )
        assert fm.metadata is not None
        assert fm.metadata["version"] == "1.2.3"

    def test_top_level_version_lifted_into_metadata(self) -> None:
        """Back-compat: top-level 'version' in YAML moves into metadata."""
        fm = SkillFrontmatter.model_validate(
            {"name": "my-skill", "description": "Desc.", "version": "0.1.0"}
        )
        assert fm.metadata is not None
        assert fm.metadata["version"] == "0.1.0"

    def test_existing_metadata_version_wins_over_top_level(self) -> None:
        fm = SkillFrontmatter.model_validate(
            {
                "name": "my-skill",
                "description": "Desc.",
                "version": "0.1.0",
                "metadata": {"version": "9.9.9"},
            }
        )
        assert fm.metadata is not None
        assert fm.metadata["version"] == "9.9.9"

    def test_existing_frontmatter_without_version_still_valid(self) -> None:
        fm = SkillFrontmatter(name="old-skill", description="No version.")
        assert fm.metadata is None