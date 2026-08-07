import pytest

from crewai.crews.crew_output import CrewOutput
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.types.usage_metrics import UsageMetrics


def test_json_raises_value_error_on_empty_tasks_output():
    """CrewOutput.json should raise the documented ValueError when there is no
    JSON output -- including when tasks_output is empty, which is a real,
    reachable state (constructed directly elsewhere in this test suite, e.g.
    test_crew.py's kickoff_for_each_async mocks), not just an empty-input
    contrivance."""
    crew_output = CrewOutput(
        raw="",
        tasks_output=[],
        token_usage=UsageMetrics(),
    )

    with pytest.raises(ValueError, match="No JSON output found"):
        _ = crew_output.json


def test_json_returns_dumped_output_when_final_task_is_json():
    task_output = TaskOutput(
        description="task",
        agent="tester",
        raw='{"answer": 42}',
        json_dict={"answer": 42},
        output_format=OutputFormat.JSON,
    )
    crew_output = CrewOutput(
        raw="",
        tasks_output=[task_output],
        token_usage=UsageMetrics(),
        json_dict={"answer": 42},
    )

    assert crew_output.json == '{"answer": 42}'
