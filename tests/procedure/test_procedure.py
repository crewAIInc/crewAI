from unittest.mock import MagicMock

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.procedure.procedure import Procedure
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


@pytest.fixture
def mock_crew_factory():
    def _create_mock_crew():
        crew = MagicMock(spec=Crew)
        task_output = TaskOutput(
            description="Test task", raw="Task output", agent="Test Agent"
        )
        crew_output = CrewOutput(
            raw="Test output",
            tasks_output=[task_output],
            token_usage={
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            },
            json_dict={"key": "value"},
        )

        async def async_kickoff(inputs=None):
            return crew_output

        crew.kickoff.return_value = crew_output
        crew.kickoff_async.side_effect = async_kickoff

        # Add more attributes that Procedure might be expecting
        crew.verbose = 0
        crew.output_log_file = None
        crew.max_rpm = None
        crew.memory = False
        crew.process = Process.sequential
        crew.config = None
        crew.cache = True

        # Add non-empty agents and tasks
        mock_agent = MagicMock(spec=Agent)
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        mock_task.async_execution = False
        mock_task.context = None

        crew.agents = [mock_agent]
        crew.tasks = [mock_task]

        return crew

    return _create_mock_crew


def test_procedure_initialization(mock_crew_factory):
    """
    Test that a Procedure is correctly initialized with the given crews.
    """
    crew1 = mock_crew_factory()
    crew2 = mock_crew_factory()

    procedure = Procedure(crews=[crew1, crew2])
    assert len(procedure.crews) == 2
    assert procedure.crews[0] == crew1
    assert procedure.crews[1] == crew2


@pytest.mark.asyncio
async def test_procedure_kickoff_single_input(mock_crew_factory):
    """
    Test that Procedure.kickoff() correctly processes a single input
    and returns the expected CrewOutput.
    """
    mock_crew_1 = mock_crew_factory()
    procedure = Procedure(crews=[mock_crew_1])
    input_data = {"key": "value"}
    result = await procedure.kickoff([input_data])

    mock_crew_1.kickoff_async.assert_called_once_with(inputs=input_data)
    assert len(result) == 1
    assert isinstance(result[0], CrewOutput)
    assert result[0].raw == "Test output"
    assert len(result[0].tasks_output) == 1
    assert result[0].tasks_output[0].raw == "Task output"
    assert result[0].token_usage == {
        "total_tokens": 100,
        "prompt_tokens": 50,
        "completion_tokens": 50,
    }


@pytest.mark.asyncio
async def test_procedure_kickoff_multiple_inputs(mock_crew_factory):
    """
    Test that Procedure.kickoff() correctly processes multiple inputs
    and returns the expected CrewOutputs.
    """
    mock_crew_1, mock_crew_2 = mock_crew_factory(), mock_crew_factory()
    procedure = Procedure(crews=[mock_crew_1, mock_crew_2])
    input_data = [{"key1": "value1"}, {"key2": "value2"}]
    result = await procedure.kickoff(input_data)

    expected_call_count_per_crew = 2
    assert mock_crew_1.kickoff_async.call_count == expected_call_count_per_crew
    assert mock_crew_2.kickoff_async.call_count == expected_call_count_per_crew
    assert len(result) == 2
    assert all(isinstance(r, CrewOutput) for r in result)
    assert all(len(r.tasks_output) == 1 for r in result)
    assert all(
        r.token_usage
        == {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}
        for r in result
    )


@pytest.mark.asyncio
async def test_procedure_chaining(mock_crew_factory):
    """
    Test that Procedure correctly chains multiple crews, passing the output
    of one crew as input to the next crew in the sequence.

    This test verifies:
    1. The first crew receives the initial input.
    2. The second crew receives the output from the first crew as its input.
    3. The final output contains the result from the last crew in the chain.
    4. Task outputs and token usage are correctly propagated through the chain.
    """
    crew1, crew2 = mock_crew_factory(), mock_crew_factory()
    task_output1 = TaskOutput(description="Task 1", raw="Output 1", agent="Agent 1")
    task_output2 = TaskOutput(description="Task 2", raw="Final output", agent="Agent 2")

    crew_output1 = CrewOutput(
        raw="Output 1",
        tasks_output=[task_output1],
        token_usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        json_dict={"key1": "value1"},
    )
    crew_output2 = CrewOutput(
        raw="Final output",
        tasks_output=[task_output2],
        token_usage={"total_tokens": 150, "prompt_tokens": 75, "completion_tokens": 75},
        json_dict={"key2": "value2"},
    )

    async def async_kickoff1(inputs=None):
        return crew_output1

    async def async_kickoff2(inputs=None):
        return crew_output2

    crew1.kickoff_async.side_effect = async_kickoff1
    crew2.kickoff_async.side_effect = async_kickoff2

    procedure = Procedure(crews=[crew1, crew2])
    input_data = [{"initial": "data"}]
    result = await procedure.kickoff(input_data)

    # Check that the first crew received the initial input
    crew1.kickoff_async.assert_called_once_with(inputs={"initial": "data"})

    # Check that the second crew received the output from the first crew as its input
    crew2.kickoff_async.assert_called_once_with(inputs=crew_output1.to_dict())

    # Check the final output
    assert len(result) == 1
    assert isinstance(result[0], CrewOutput)
    assert result[0].raw == "Final output"
    assert len(result[0].tasks_output) == 1
    assert result[0].tasks_output[0].raw == "Final output"
    assert result[0].token_usage == {
        "total_tokens": 150,
        "prompt_tokens": 75,
        "completion_tokens": 75,
    }
    assert result[0].json_dict == {"key2": "value2"}
