"""Tests for self_improve/acceptance.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from crewai.skills.self_improve.acceptance import (
    _format_skill_md,
    accept_proposal,
    reject_proposal,
)
from crewai.skills.self_improve.models import SkillProposal
from crewai.skills.self_improve.storage import ProposalStore, SkillStore


@pytest.fixture
def proposal() -> SkillProposal:
    return SkillProposal(
        agent_role="Senior Researcher",
        name="cite-sources",
        description="Always cite sources in research outputs",
        body="# Cite Sources\n\n*Always cite sources.*\n\n## When to use\n\nWhenever you write a research summary.\n",
        rationale="Seen in 3 of 4 traces",
        confidence=0.8,
        derived_from_runs=["run_a", "run_b", "run_c"],
    )


@pytest.fixture
def stores(tmp_path: Path):
    return SkillStore(root=tmp_path), ProposalStore(root=tmp_path)


class TestFormatSkillMd:
    def test_includes_yaml_frontmatter(self, proposal: SkillProposal) -> None:
        out = _format_skill_md(proposal)
        assert out.startswith("---\n")
        # Values are JSON-quoted (valid YAML scalars).
        assert 'name: "cite-sources"' in out
        assert '"Always cite sources in research outputs"' in out
        assert "# Cite Sources" in out

    def test_quotes_descriptions_with_special_chars(self) -> None:
        prop = SkillProposal(
            agent_role="r",
            name="n",
            description='Has a "quote" and: colon',
            body="b",
            rationale="r",
            confidence=0.7,
        )
        out = _format_skill_md(prop)
        # JSON-quoted: backslash-escapes the inner quote, retains the colon literally.
        assert 'description: "Has a \\"quote\\" and: colon"' in out

    def test_passes_through_body_with_existing_frontmatter(self) -> None:
        prop = SkillProposal(
            agent_role="r",
            name="n",
            description="d",
            body="---\nname: n\n---\n# Body\n",
            rationale="r",
            confidence=0.7,
        )
        out = _format_skill_md(prop)
        # No double frontmatter
        assert out.count("---") == 2  # one open, one close


class TestAcceptProposal:
    def test_writes_skill_md_at_expected_path(
        self, stores, proposal: SkillProposal
    ) -> None:
        skill_store, proposal_store = stores
        proposal_store.save(proposal)

        path = accept_proposal(
            proposal,
            skill_store=skill_store,
            proposal_store=proposal_store,
        )

        assert path.name == "SKILL.md"
        # role gets slugified
        assert "senior-researcher" in str(path)
        assert "cite-sources" in str(path)

        body = path.read_text()
        assert body.startswith("---\n")
        assert "# Cite Sources" in body

    def test_removes_proposal_from_queue_on_accept(
        self, stores, proposal: SkillProposal
    ) -> None:
        skill_store, proposal_store = stores
        proposal_store.save(proposal)
        assert proposal_store.find(proposal.id) is not None

        accept_proposal(
            proposal, skill_store=skill_store, proposal_store=proposal_store
        )

        assert proposal_store.find(proposal.id) is None

    def test_refuses_to_overwrite_existing(
        self, stores, proposal: SkillProposal
    ) -> None:
        skill_store, proposal_store = stores
        proposal_store.save(proposal)
        accept_proposal(
            proposal, skill_store=skill_store, proposal_store=proposal_store
        )
        # Re-save and re-accept the same proposal id (or a fresh one) → conflict
        proposal_store.save(proposal)
        with pytest.raises(FileExistsError):
            accept_proposal(
                proposal,
                skill_store=skill_store,
                proposal_store=proposal_store,
            )

    def test_force_overwrites(self, stores, proposal: SkillProposal) -> None:
        skill_store, proposal_store = stores
        proposal_store.save(proposal)
        accept_proposal(
            proposal, skill_store=skill_store, proposal_store=proposal_store
        )

        proposal_store.save(proposal)
        path = accept_proposal(
            proposal,
            skill_store=skill_store,
            proposal_store=proposal_store,
            force=True,
        )
        assert path.exists()


class TestRejectProposal:
    def test_removes_from_queue(self, stores, proposal: SkillProposal) -> None:
        _, proposal_store = stores
        proposal_store.save(proposal)
        assert reject_proposal(proposal, proposal_store=proposal_store) is True
        assert proposal_store.find(proposal.id) is None

    def test_returns_false_when_missing(
        self, stores, proposal: SkillProposal
    ) -> None:
        _, proposal_store = stores
        assert reject_proposal(proposal, proposal_store=proposal_store) is False


class TestSkillStore:
    def test_has_any_detects_at_least_one_skill(
        self, stores, proposal: SkillProposal
    ) -> None:
        skill_store, proposal_store = stores
        assert skill_store.has_any("Senior Researcher") is False
        proposal_store.save(proposal)
        accept_proposal(
            proposal, skill_store=skill_store, proposal_store=proposal_store
        )
        assert skill_store.has_any("Senior Researcher") is True
