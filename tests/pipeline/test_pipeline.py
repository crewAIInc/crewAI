from unittest.mock import MagicMock

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.pipeline.pipeline import Pipeline
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


@pytest.fixture
def mock_crew_factory():
    def _create_mock_crew(output_json_dict=None):
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
            json_dict=output_json_dict if output_json_dict else {"key": "value"},
        )

        async def async_kickoff(inputs=None):
            print("inputs in async_kickoff", inputs)
            return crew_output

        crew.kickoff_async.side_effect = async_kickoff

        # Add more attributes that Procedure might be expecting
        crew.verbose = False
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


def test_pipeline_initialization(mock_crew_factory):
    """
    Test that a Pipeline is correctly initialized with the given stages.
    """
    crew1 = mock_crew_factory()
    crew2 = mock_crew_factory()

    pipeline = Pipeline(stages=[crew1, crew2])
    assert len(pipeline.stages) == 2
    assert pipeline.stages[0] == crew1
    assert pipeline.stages[1] == crew2


@pytest.mark.asyncio
async def test_pipeline_process_streams_single_input(mock_crew_factory):
    """
    Test that Pipeline.process_streams() correctly processes a single input
    and returns the expected CrewOutput.
    """
    mock_crew = mock_crew_factory()
    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key": "value"}]
    pipeline_result = await pipeline.process_streams(input_data)

    mock_crew.kickoff_async.assert_called_once_with(inputs={"key": "value"})
    for stream_result in pipeline_result:
        assert isinstance(stream_result[0], CrewOutput)
        assert stream_result[0].raw == "Test output"
        assert len(stream_result[0].tasks_output) == 1
        assert stream_result[0].tasks_output[0].raw == "Task output"
        assert stream_result[0].token_usage == {
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
        }


@pytest.mark.asyncio
async def test_pipeline_process_streams_multiple_inputs(mock_crew_factory):
    """
    Test that Pipeline.process_streams() correctly processes multiple inputs
    and returns the expected CrewOutputs.
    """
    mock_crew = mock_crew_factory()
    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key1": "value1"}, {"key2": "value2"}]
    pipeline_result = await pipeline.process_streams(input_data)

    assert mock_crew.kickoff_async.call_count == 2
    assert len(pipeline_result) == 2
    for stream_result in pipeline_result:
        assert all(
            isinstance(stream_output, CrewOutput) for stream_output in stream_result
        )


@pytest.mark.asyncio
async def test_pipeline_with_parallel_stages(mock_crew_factory):
    """
    Test that Pipeline correctly handles parallel stages.
    """
    crew1 = mock_crew_factory()
    crew2 = mock_crew_factory()
    crew3 = mock_crew_factory()

    pipeline = Pipeline(stages=[crew1, [crew2, crew3]])
    input_data = [{"initial": "data"}]

    pipeline_result = await pipeline.process_streams(input_data)

    crew1.kickoff_async.assert_called_once_with(
        inputs={"initial": "data", "key": "value"}
    )
    crew2.kickoff_async.assert_called_once_with(
        inputs={"initial": "data", "key": "value"}
    )
    crew3.kickoff_async.assert_called_once_with(
        inputs={"initial": "data", "key": "value"}
    )

    assert len(pipeline_result) == 1
    for stage_result in pipeline_result:
        assert isinstance(stage_result[0], CrewOutput)


def test_pipeline_rshift_operator(mock_crew_factory):
    """
    Test that the >> operator correctly creates a Pipeline from Crews and lists of Crews.
    """
    crew1 = mock_crew_factory()
    crew2 = mock_crew_factory()
    crew3 = mock_crew_factory()

    # Test single crew addition
    pipeline = Pipeline(stages=[]) >> crew1
    assert len(pipeline.stages) == 1
    assert pipeline.stages[0] == crew1

    # Test adding a list of crews
    pipeline = Pipeline(stages=[crew1])
    pipeline = pipeline >> [crew2, crew3]
    print("pipeline.stages:", pipeline.stages)
    assert len(pipeline.stages) == 2
    assert pipeline.stages[1] == [crew2, crew3]

    # Test error case: trying to shift with non-Crew object
    with pytest.raises(TypeError):
        pipeline >> "not a crew"


"""
TODO: Figure out what is the proper output for a pipeline with multiple stages

Options:
- Should the final output only include the last stage's output?
- Should the final output include the accumulation of previous stages' outputs?

"""


@pytest.mark.asyncio
async def test_pipeline_data_accumulation(mock_crew_factory):
    """
    Test that data is correctly accumulated through the pipeline stages.
    """
    crew1 = mock_crew_factory(output_json_dict={"key1": "value1"})
    crew2 = mock_crew_factory(output_json_dict={"key2": "value2"})

    pipeline = Pipeline(stages=[crew1, crew2])
    input_data = [{"initial": "data"}]
    pipeline_result = await pipeline.process_streams(input_data)

    assert len(pipeline_result) == 1
    print("RESULT: ", pipeline_result)
    for stream_result in pipeline_result:
        print("STREAM RESULT: ", stream_result)
        assert stream_result[0].json_dict == {
            "initial": "data",
            "key1": "value1",
            "key2": "value2",
        }
