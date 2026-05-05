"""Tests for self_improve/models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pathlib import Path

from crewai.skills.self_improve.models import (
    RunTrace,
    SelfImprovementConfig,
    SkillProposal,
    ToolCallRecord,
)


class TestSelfImprovementConfig:
    def test_defaults(self) -> None:
        cfg = SelfImprovementConfig()
        assert cfg.skills_dir is None

    def test_skills_dir_round_trip(self, tmp_path: Path) -> None:
        cfg = SelfImprovementConfig(skills_dir=tmp_path / "learned")
        assert cfg.skills_dir == tmp_path / "learned"


class TestRunTrace:
    def test_minimal_trace_has_defaults(self) -> None:
        trace = RunTrace(agent_role="researcher")
        assert trace.agent_role == "researcher"
        assert trace.outcome == "unknown"
        assert trace.tool_calls == []
        assert trace.id.startswith("run_")
        assert trace.tool_call_count == 0
        assert trace.tool_error_count == 0

    def test_tool_counters(self) -> None:
        trace = RunTrace(
            agent_role="researcher",
            tool_calls=[
                ToolCallRecord(name="search", ok=True),
                ToolCallRecord(name="search", ok=False, error="boom"),
            ],
        )
        assert trace.tool_call_count == 2
        assert trace.tool_error_count == 1

    def test_serializes_roundtrip(self) -> None:
        trace = RunTrace(
            agent_role="researcher",
            tool_calls=[ToolCallRecord(name="search", args_summary="q=hi")],
        )
        payload = trace.model_dump_json()
        roundtrip = RunTrace.model_validate_json(payload)
        assert roundtrip.id == trace.id
        assert roundtrip.tool_calls[0].name == "search"


class TestSkillProposal:
    def test_minimal_proposal(self) -> None:
        prop = SkillProposal(
            agent_role="researcher",
            name="my-skill",
            description="A skill",
            body="# body",
            rationale="seen 3 times",
            confidence=0.8,
        )
        assert prop.proposal_kind == "new"
        assert prop.id.startswith("prop_")

    def test_confidence_must_be_in_range(self) -> None:
        with pytest.raises(ValidationError):
            SkillProposal(
                agent_role="r",
                name="n",
                description="d",
                body="b",
                rationale="r",
                confidence=1.5,
            )
