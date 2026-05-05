"""Tests for self_improve/auto_grade.py."""

from __future__ import annotations

from crewai.skills.self_improve.auto_grade import grade_trace
from crewai.skills.self_improve.models import RunTrace, ToolCallRecord


def _trace(**kw):
    return RunTrace(agent_role="r", **kw)


def test_explicit_error_is_failure() -> None:
    assert grade_trace(_trace(error="kaboom", output_summary="ok")) == "failure"


def test_guardrail_pass_overrides_other_signals() -> None:
    trace = _trace(
        guardrail_passed=True,
        max_iter_exhausted=True,  # would normally fail, but guardrail wins
        output_summary="ok",
    )
    assert grade_trace(trace) == "success"


def test_guardrail_fail_is_failure() -> None:
    assert grade_trace(_trace(guardrail_passed=False, output_summary="x")) == "failure"


def test_max_iter_is_failure() -> None:
    assert grade_trace(_trace(max_iter_exhausted=True, output_summary="x")) == "failure"


def test_thrashing_is_failure() -> None:
    trace = _trace(
        tool_calls=[
            ToolCallRecord(name="search", args_summary="q=x") for _ in range(5)
        ],
        output_summary="ok",
    )
    assert grade_trace(trace) == "failure"


def test_empty_output_is_failure() -> None:
    assert grade_trace(_trace(output_summary="   ")) == "failure"


def test_error_string_output_is_failure() -> None:
    assert grade_trace(_trace(output_summary="Error: boom")) == "failure"


def test_minority_tool_errors_still_count_as_success() -> None:
    trace = _trace(
        tool_calls=[
            ToolCallRecord(name="a", ok=True),
            ToolCallRecord(name="b", ok=True),
            ToolCallRecord(name="c", ok=False, error="x"),
        ],
        output_summary="answer",
    )
    assert grade_trace(trace) == "success"


def test_failure_when_majority_tool_errors() -> None:
    trace = _trace(
        tool_calls=[
            ToolCallRecord(name="a", ok=False, error="x"),
            ToolCallRecord(name="b", ok=False, error="x"),
            ToolCallRecord(name="c", ok=True),
        ],
        output_summary="answer",
    )
    assert grade_trace(trace) == "failure"


def test_clean_run_is_success() -> None:
    trace = _trace(
        tool_calls=[ToolCallRecord(name="a", ok=True)],
        output_summary="answer",
    )
    assert grade_trace(trace) == "success"


def test_no_signal_is_unknown() -> None:
    assert grade_trace(_trace()) == "unknown"
