"""Tests for the 'version' metadata key on SkillFrontmatter.

Per the agentskills.io spec, `version` lives under `metadata`, not as a
top-level frontmatter field.
"""

from __future__ import annotations

from crewai.skills.models import SkillFrontmatter


class TestSkillFrontmatterVersion:
    def test_no_metadata_by_default(self) -> None:
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

    def test_metadata_accepts_other_keys(self) -> None:
        fm = SkillFrontmatter(
            name="my-skill",
            description="A skill.",
            metadata={"version": "1.0.0", "author": "acme"},
        )
        assert fm.metadata == {"version": "1.0.0", "author": "acme"}
