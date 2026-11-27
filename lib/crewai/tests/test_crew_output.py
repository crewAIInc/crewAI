import pytest

from crewai.crews.crew_output import CrewOutput


def test_json_raises_value_error_when_tasks_output_is_empty():
    """A CrewOutput built outside a kickoff (empty tasks_output) must raise the
    guided ValueError instead of a bare IndexError."""
    output = CrewOutput(raw="no tasks")

    with pytest.raises(ValueError, match="No JSON output"):
        _ = output.json
