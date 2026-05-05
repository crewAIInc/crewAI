"""Tests for ``crewai skills proposals`` CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
import pytest

from crewai.cli.skills_proposals import skills as skills_group
from crewai.skills.self_improve.models import SkillProposal
from crewai.skills.self_improve.storage import ProposalStore, SkillStore


@pytest.fixture
def proposal() -> SkillProposal:
    return SkillProposal(
        agent_role="researcher",
        name="cite-sources",
        description="Always cite sources",
        body="# Cite Sources\n\nbody.",
        rationale="seen in 3 traces",
        confidence=0.75,
    )


@pytest.fixture
def runner_with_root(tmp_path: Path):
    """CliRunner with the self-improve root patched at the storage layer."""
    runner = CliRunner()
    proposal_store = ProposalStore(root=tmp_path)
    skill_store = SkillStore(root=tmp_path)

    with patch(
        "crewai.cli.skills_proposals.ProposalStore", return_value=proposal_store
    ), patch(
        "crewai.skills.self_improve.acceptance.ProposalStore",
        return_value=proposal_store,
    ), patch(
        "crewai.skills.self_improve.acceptance.SkillStore", return_value=skill_store
    ):
        yield runner, proposal_store, skill_store


class TestList:
    def test_empty(self, runner_with_root) -> None:
        runner, _, _ = runner_with_root
        result = runner.invoke(skills_group, ["proposals", "list"])
        assert result.exit_code == 0
        assert "no pending proposals" in result.output

    def test_one_pending(self, runner_with_root, proposal: SkillProposal) -> None:
        runner, ps, _ = runner_with_root
        ps.save(proposal)
        result = runner.invoke(skills_group, ["proposals", "list"])
        assert result.exit_code == 0
        assert proposal.id in result.output
        assert "cite-sources" in result.output


class TestShow:
    def test_unknown_id_exits_nonzero(self, runner_with_root) -> None:
        runner, _, _ = runner_with_root
        result = runner.invoke(skills_group, ["proposals", "show", "prop_does_not_exist"])
        assert result.exit_code == 1
        assert "No proposal" in result.output

    def test_prints_body(self, runner_with_root, proposal: SkillProposal) -> None:
        runner, ps, _ = runner_with_root
        ps.save(proposal)
        result = runner.invoke(skills_group, ["proposals", "show", proposal.id])
        assert result.exit_code == 0
        assert "# Cite Sources" in result.output
        assert "rationale" in result.output


class TestAccept:
    def test_writes_skill_md_and_clears_queue(
        self, runner_with_root, proposal: SkillProposal
    ) -> None:
        runner, ps, ss = runner_with_root
        ps.save(proposal)

        result = runner.invoke(skills_group, ["proposals", "accept", proposal.id])
        assert result.exit_code == 0, result.output
        assert ps.find(proposal.id) is None
        assert (ss.skill_dir("researcher", "cite-sources") / "SKILL.md").is_file()

    def test_unknown_id_exits_nonzero(self, runner_with_root) -> None:
        runner, _, _ = runner_with_root
        result = runner.invoke(skills_group, ["proposals", "accept", "prop_nope"])
        assert result.exit_code == 1


class TestReject:
    def test_removes_from_queue(self, runner_with_root, proposal: SkillProposal) -> None:
        runner, ps, _ = runner_with_root
        ps.save(proposal)

        result = runner.invoke(skills_group, ["proposals", "reject", proposal.id])
        assert result.exit_code == 0
        assert ps.find(proposal.id) is None
