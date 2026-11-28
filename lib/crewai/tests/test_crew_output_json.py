"""Regression tests for CrewOutput.json with skipped conditional tasks."""
import pytest

from crewai.crews.crew_output import CrewOutput
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.usage_metrics import UsageMetrics


def _usage():
    return UsageMetrics(
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        successful_requests=0,
    )


def test_json_returns_value_when_last_task_is_skipped():
    """CrewOutput.json should use the last non-empty task when the final task was skipped."""
    produced = TaskOutput(
        description="Produce JSON",
        agent="Writer",
        raw='{"answer": 42}',
        output_format=OutputFormat.JSON,
        json_dict={"answer": 42},
    )
    skipped = TaskOutput(description="Skipped", agent="Reviewer", raw="")
    result = CrewOutput(
        raw=produced.raw,
        json_dict=produced.json_dict,
        tasks_output=[produced, skipped],
        token_usage=_usage(),
    )
    assert result.json == '{"answer": 42}'


def test_json_raises_when_no_task_produced_json():
    """CrewOutput.json should still raise when no task produced JSON output."""
    plain = TaskOutput(
        description="Plain task",
        agent="Writer",
        raw="just text",
        output_format=OutputFormat.RAW,
    )
    result = CrewOutput(
        raw=plain.raw,
        json_dict=None,
        tasks_output=[plain],
        token_usage=_usage(),
    )
    with pytest.raises(ValueError, match="No JSON output found"):
        _ = result.json
