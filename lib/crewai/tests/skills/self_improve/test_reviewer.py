"""Tests for self_improve/reviewer.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.skills.self_improve.models import RunTrace, SkillProposal, ToolCallRecord
from crewai.skills.self_improve.reviewer import (
    SkillReviewer,
    _ReviewerOutput,
    _format_trace,
)


def _llm_proposal(**kw):
    """Build a SkillProposal as the LLM would emit it (no server-filled fields)."""
    base = dict(
        name="x",
        description="d",
        body="b",
        rationale="r",
        confidence=0.7,
    )
    base.update(kw)
    return SkillProposal(**base)


def _trace(**kw: Any) -> RunTrace:
    base = {"agent_role": "researcher", "outcome": "success"}
    base.update(kw)
    return RunTrace(**base)


def _stub_llm(output: _ReviewerOutput) -> MagicMock:
    """Return a BaseLLM stand-in whose ``call`` returns the given output."""
    llm = MagicMock()
    llm.call = MagicMock(return_value=output)
    return llm


@pytest.fixture
def reviewer_factory():
    def make(llm, **kw):
        return SkillReviewer(
            agent_role="researcher",
            agent_goal="answer questions",
            llm=llm,
            **kw,
        )

    return make


class TestFormatTrace:
    def test_includes_outcome_task_and_tool_calls(self) -> None:
        trace = _trace(
            task_description="Find papers on X",
            output_summary="Found 3 papers.",
            tool_calls=[
                ToolCallRecord(name="search", args_summary="q=X", ok=True),
                ToolCallRecord(name="fetch", args_summary="id=42", ok=False, error="404"),
            ],
        )
        block = _format_trace(trace)
        assert "outcome=success" in block
        assert "Find papers on X" in block
        assert "Found 3 papers." in block
        assert "[ok] search(q=X)" in block
        assert "[ERR] fetch(id=42)" in block

    def test_truncates_long_output(self) -> None:
        trace = _trace(output_summary="x" * 5000)
        block = _format_trace(trace)
        assert "…" in block
        assert len(block) < 5000


class TestSkillReviewer:
    def test_returns_empty_when_below_min_traces(self, reviewer_factory) -> None:
        llm = _stub_llm(_ReviewerOutput(proposals=[]))
        reviewer = reviewer_factory(llm, min_traces=3)
        result = reviewer.review([_trace(), _trace()])
        assert result == []
        llm.call.assert_not_called()

    def test_filters_by_confidence_floor(self, reviewer_factory) -> None:
        llm_output = _ReviewerOutput(
            proposals=[
                _llm_proposal(name="keep", confidence=0.8),
                _llm_proposal(name="drop", confidence=0.3),
            ]
        )
        llm = _stub_llm(llm_output)
        reviewer = reviewer_factory(llm, min_traces=2, confidence_floor=0.6)
        out = reviewer.review([_trace(), _trace(), _trace()])
        assert [p.name for p in out] == ["keep"]

    def test_sets_agent_role_and_run_ids(self, reviewer_factory) -> None:
        llm = _stub_llm(
            _ReviewerOutput(proposals=[_llm_proposal(name="cite-sources")])
        )
        reviewer = reviewer_factory(llm, min_traces=2)
        traces = [_trace(), _trace(), _trace()]
        out = reviewer.review(traces)
        assert len(out) == 1
        prop = out[0]
        assert prop.agent_role == "researcher"
        assert prop.derived_from_runs == [t.id for t in traces]
        assert prop.proposal_kind == "new"

    def test_passes_loaded_skills_into_prompt(self, reviewer_factory) -> None:
        llm = _stub_llm(_ReviewerOutput(proposals=[]))
        reviewer = reviewer_factory(llm, min_traces=2)
        reviewer.review(
            [_trace(), _trace(), _trace()],
            loaded_skill_names=["citing", "search-tactics"],
        )
        call_kwargs = llm.call.call_args.kwargs
        messages = call_kwargs["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "citing" in system_msg
        assert "search-tactics" in system_msg

    def test_handles_non_model_response_gracefully(self, reviewer_factory) -> None:
        # An LLM that returned a plain string instead of the structured model.
        llm = MagicMock()
        llm.call = MagicMock(return_value="totally not a model")
        reviewer = reviewer_factory(llm, min_traces=2)
        out = reviewer.review([_trace(), _trace(), _trace()])
        assert out == []

    def test_pending_proposals_appear_in_prompt(self, reviewer_factory) -> None:
        llm = _stub_llm(_ReviewerOutput(proposals=[]))
        reviewer = reviewer_factory(llm, min_traces=2)
        pending = [
            SkillProposal(
                agent_role="researcher",
                name="cite-sources",
                description="Always cite sources in research output.",
                body="b",
                rationale="r",
                confidence=0.8,
            )
        ]
        reviewer.review(
            [_trace(), _trace(), _trace()],
            pending_proposals=pending,
        )
        messages = llm.call.call_args.kwargs["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "cite-sources" in system_msg
        assert "Always cite sources" in system_msg
        # And it should be in the queued-proposals section, not the loaded
        # skills section.
        assert "QUEUED" in system_msg

    def test_pending_proposals_default_none_renders_none(
        self, reviewer_factory
    ) -> None:
        llm = _stub_llm(_ReviewerOutput(proposals=[]))
        reviewer = reviewer_factory(llm, min_traces=2)
        reviewer.review([_trace(), _trace(), _trace()])
        system_msg = next(
            m["content"]
            for m in llm.call.call_args.kwargs["messages"]
            if m["role"] == "system"
        )
        # The "(none)" sentinel renders under both sections when nothing was passed.
        assert system_msg.count("(none)") >= 2

    def test_patch_existing_kind_passes_through(self, reviewer_factory) -> None:
        llm = _stub_llm(
            _ReviewerOutput(
                proposals=[
                    _llm_proposal(
                        name="citing-v2",
                        confidence=0.9,
                        proposal_kind="patch_existing",
                        target_skill="citing",
                    )
                ]
            )
        )
        reviewer = reviewer_factory(llm, min_traces=2)
        [prop] = reviewer.review(
            [_trace(), _trace(), _trace()], loaded_skill_names=["citing"]
        )
        assert prop.proposal_kind == "patch_existing"
        assert prop.target_skill == "citing"
