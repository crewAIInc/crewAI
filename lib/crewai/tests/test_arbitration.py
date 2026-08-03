"""Tests for deterministic arbitration / escrow guardrails."""

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from crewai import Agent, Task
from crewai.tasks.arbitration import (
    ArbitrationEngine,
    ArbitrationGuardrail,
    ArbitrationStatus,
)
from crewai.tasks.task_output import TaskOutput


class CodeReviewOutputContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str
    summary: str = Field(min_length=20, max_length=500)
    confidence_score: float = Field(ge=0.0, le=1.0)
    files_reviewed: list[str] = Field(min_length=1)
    blocking_issues_found: int = Field(ge=0)

    @field_validator("status")
    @classmethod
    def status_must_be_allowed(cls, value: str) -> str:
        allowed = {"completed", "needs_changes", "escalated"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value


@pytest.fixture
def engine() -> ArbitrationEngine:
    return ArbitrationEngine()


@pytest.fixture
def valid_payload() -> dict:
    return {
        "task_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "completed",
        "summary": "Reviewed PR #482: found two minor style nits, no blocking issues.",
        "confidence_score": 0.92,
        "files_reviewed": ["src/api/routes.py", "src/api/models.py"],
        "blocking_issues_found": 0,
    }


def test_evaluate_approves_valid_payload(
    engine: ArbitrationEngine, valid_payload: dict
) -> None:
    result = engine.evaluate(CodeReviewOutputContract, valid_payload)

    assert result.status is ArbitrationStatus.APPROVED
    assert result.is_approved
    assert result.violations == []
    assert result.validated_payload == valid_payload


def test_evaluate_disputes_invalid_payload(engine: ArbitrationEngine) -> None:
    bad_payload = {
        "task_id": "not-a-uuid",
        "status": "done",
        "summary": "too short",
        "confidence_score": 1.4,
        "files_reviewed": [],
        "blocking_issues_found": -3,
        "unexpected_field": "hack attempt",
    }

    result = engine.evaluate(CodeReviewOutputContract, bad_payload)

    assert result.status is ArbitrationStatus.DISPUTED
    assert not result.is_approved
    assert len(result.violations) >= 1
    instructions = result.to_retry_instructions()
    assert "DISPUTED" in instructions
    assert "constraint=" in instructions


def test_evaluate_disputes_invalid_json(engine: ArbitrationEngine) -> None:
    result = engine.evaluate(CodeReviewOutputContract, "not-json")

    assert result.status is ArbitrationStatus.DISPUTED
    assert result.violations[0].constraint == "invalid_json"


def test_evaluate_disputes_missed_deadline(
    engine: ArbitrationEngine, valid_payload: dict
) -> None:
    deadline = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evaluated_at = deadline + timedelta(minutes=5)

    result = engine.evaluate(
        CodeReviewOutputContract,
        valid_payload,
        deadline=deadline,
        evaluated_at=evaluated_at,
    )

    assert result.status is ArbitrationStatus.DISPUTED
    assert any(v.constraint == "deadline_missed" for v in result.violations)


def test_evaluate_cel_rule_failure(
    engine: ArbitrationEngine, valid_payload: dict
) -> None:
    result = engine.evaluate(
        CodeReviewOutputContract,
        valid_payload,
        rules=["output.confidence_score >= 0.95"],
    )

    assert result.status is ArbitrationStatus.DISPUTED
    assert any(v.constraint == "rule_failed" for v in result.violations)


def test_evaluate_cel_rule_success(
    engine: ArbitrationEngine, valid_payload: dict
) -> None:
    result = engine.evaluate(
        CodeReviewOutputContract,
        valid_payload,
        rules=[
            "output.confidence_score >= 0.9",
            "output.blocking_issues_found == 0",
        ],
    )

    assert result.status is ArbitrationStatus.APPROVED


def test_evaluate_accepts_task_output_json_dict(
    engine: ArbitrationEngine, valid_payload: dict
) -> None:
    task_output = TaskOutput(
        description="Review code",
        agent="reviewer",
        raw="ignored when json_dict is set",
        json_dict=valid_payload,
    )

    result = engine.evaluate(CodeReviewOutputContract, task_output)

    assert result.is_approved


def test_guardrail_call_approves_and_returns_json(valid_payload: dict) -> None:
    guardrail = ArbitrationGuardrail(CodeReviewOutputContract)
    task_output = TaskOutput(
        description="Review code",
        agent="reviewer",
        raw=str(valid_payload),
        json_dict=valid_payload,
    )

    ok, value = guardrail(task_output)

    assert ok is True
    assert '"status": "completed"' in value
    assert "ArbitrationGuardrail(CodeReviewOutputContract)" in guardrail.description


def test_guardrail_call_disputes_with_retry_instructions() -> None:
    guardrail = ArbitrationGuardrail(CodeReviewOutputContract)
    task_output = TaskOutput(
        description="Review code",
        agent="reviewer",
        raw='{"status": "done"}',
    )

    ok, feedback = guardrail(task_output)

    assert ok is False
    assert "DISPUTED" in feedback


def test_task_retries_when_arbitration_guardrail_fails(
    valid_payload: dict,
) -> None:
    guardrail = ArbitrationGuardrail(CodeReviewOutputContract)

    agent = Mock()
    agent.role = "reviewer"
    agent.crew = None
    agent.last_messages = []
    agent.execute_task.side_effect = [
        '{"status": "done"}',
        json.dumps(valid_payload),
    ]

    task = Task(
        description="Review the pull request",
        expected_output="Structured code review JSON",
        agent=Agent(role="reviewer", goal="review", backstory="reviews code"),
        guardrail=guardrail,
        guardrail_max_retries=1,
    )

    result = task.execute_sync(agent=agent)

    assert '"status": "completed"' in result.raw
    assert agent.execute_task.call_count == 2
