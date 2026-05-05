"""Tests for self_improve/storage.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from crewai.skills.self_improve.models import RunTrace, SkillProposal
from crewai.skills.self_improve.storage import ProposalStore, TraceStore


@pytest.fixture
def trace() -> RunTrace:
    return RunTrace(agent_role="Senior Researcher", output_summary="hello")


@pytest.fixture
def proposal() -> SkillProposal:
    return SkillProposal(
        agent_role="Senior Researcher",
        name="cite-sources",
        description="Always cite sources",
        body="# body",
        rationale="3 of 5 runs cited",
        confidence=0.7,
    )


class TestTraceStore:
    def test_save_and_load_roundtrip(self, tmp_path: Path, trace: RunTrace) -> None:
        store = TraceStore(root=tmp_path)
        path = store.save(trace)
        assert path.exists()
        # role gets slugified into the dir name
        assert "senior-researcher" in str(path)

        loaded = store.load(path)
        assert loaded.id == trace.id
        assert loaded.output_summary == "hello"

    def test_list_for_role(self, tmp_path: Path) -> None:
        store = TraceStore(root=tmp_path)
        for _ in range(3):
            store.save(RunTrace(agent_role="researcher"))
        assert store.count_for_role("researcher") == 3

    def test_role_slug_is_filesystem_safe(self, tmp_path: Path) -> None:
        store = TraceStore(root=tmp_path)
        store.save(RunTrace(agent_role="Weird/Role:Name!"))
        # only safe chars survive after slugify
        assert any(p.is_dir() for p in store.root.iterdir())


class TestProposalStore:
    def test_save_and_load_roundtrip(
        self, tmp_path: Path, proposal: SkillProposal
    ) -> None:
        store = ProposalStore(root=tmp_path)
        path = store.save(proposal)
        loaded = store.load(path)
        assert loaded.id == proposal.id
        assert loaded.name == "cite-sources"

    def test_delete(self, tmp_path: Path, proposal: SkillProposal) -> None:
        store = ProposalStore(root=tmp_path)
        store.save(proposal)
        assert store.delete(proposal.id, "Senior Researcher") is True
        assert store.delete(proposal.id, "Senior Researcher") is False
