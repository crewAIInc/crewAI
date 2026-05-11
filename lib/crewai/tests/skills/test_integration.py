"""Integration tests for the skills system."""

from pathlib import Path

import pytest

from crewai import Agent
from crewai.agent.utils import append_skill_context
from crewai.skills.loader import activate_skill, discover_skills, format_skill_context
from crewai.skills.models import INSTRUCTIONS, METADATA


def _create_skill_dir(parent: Path, name: str, body: str = "Body.") -> Path:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = parent / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Skill {name}\n---\n{body}"
    )
    return skill_dir


class TestSkillDiscoveryAndActivation:
    """End-to-end tests for discover + activate workflow."""

    def test_discover_and_activate(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "my-skill", body="Use this skill.")
        skills = discover_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].disclosure_level == METADATA

        activated = activate_skill(skills[0])
        assert activated.disclosure_level == INSTRUCTIONS
        assert activated.instructions == "Use this skill."

        context = format_skill_context(activated)
        assert "## Skill: my-skill" in context
        assert "Use this skill." in context

    def test_filter_by_skill_names(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "alpha")
        _create_skill_dir(tmp_path, "beta")
        _create_skill_dir(tmp_path, "gamma")

        all_skills = discover_skills(tmp_path)
        wanted = {"alpha", "gamma"}
        filtered = [s for s in all_skills if s.name in wanted]
        assert {s.name for s in filtered} == {"alpha", "gamma"}

    def test_full_fixture_skill(self) -> None:
        fixtures = Path(__file__).parent / "fixtures"
        valid_dir = fixtures / "valid-skill"
        if not valid_dir.exists():
            pytest.skip("Fixture not found")

        skills = discover_skills(fixtures)
        valid_skills = [s for s in skills if s.name == "valid-skill"]
        assert len(valid_skills) == 1

        skill = valid_skills[0]
        assert skill.frontmatter.license == "Apache-2.0"
        assert skill.frontmatter.allowed_tools == ["web-search", "file-read"]

        activated = activate_skill(skill)
        assert "Instructions" in (activated.instructions or "")

    def test_multiple_search_paths(self, tmp_path: Path) -> None:
        path_a = tmp_path / "a"
        path_a.mkdir()
        _create_skill_dir(path_a, "skill-a")

        path_b = tmp_path / "b"
        path_b.mkdir()
        _create_skill_dir(path_b, "skill-b")

        all_skills = []
        for search_path in [path_a, path_b]:
            all_skills.extend(discover_skills(search_path))
        names = {s.name for s in all_skills}
        assert names == {"skill-a", "skill-b"}

    def test_agent_preserves_metadata_for_discovered_skills(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "travel", body="Use this skill for travel planning.")
        discovered = discover_skills(tmp_path)

        agent = Agent(
            role="Travel Advisor",
            goal="Provide personalized travel suggestions.",
            backstory="An experienced travel consultant.",
            skills=discovered,
        )

        assert agent.skills is not None
        assert agent.skills[0].disclosure_level == METADATA
        assert agent.skills[0].instructions is None

        prompt = append_skill_context(agent, "Plan a 10-day Japan itinerary.")
        assert "## Skill: travel" in prompt
        assert "Skill travel" in prompt
        assert "Use this skill for travel planning." not in prompt
