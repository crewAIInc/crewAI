import json
from unittest.mock import MagicMock

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.pipeline.pipeline import Pipeline
from crewai.pipeline.pipeline_kickoff_result import PipelineKickoffResult
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.types.usage_metrics import UsageMetrics
from pydantic import BaseModel, ValidationError

DEFAULT_TOKEN_USAGE = UsageMetrics(
    total_tokens=100, prompt_tokens=50, completion_tokens=50, successful_requests=3
)


@pytest.fixture
def mock_crew_factory():
    def _create_mock_crew(name: str, output_json_dict=None, pydantic_output=None):
        crew = MagicMock(spec=Crew)
        task_output = TaskOutput(
            description="Test task", raw="Task output", agent="Test Agent"
        )
        crew_output = CrewOutput(
            raw="Test output",
            tasks_output=[task_output],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict=output_json_dict if output_json_dict else None,
            pydantic=pydantic_output,
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
        crew.name = name

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
    crew1 = mock_crew_factory(name="Crew 1")
    crew2 = mock_crew_factory(name="Crew 2")

    pipeline = Pipeline(stages=[crew1, crew2])
    assert len(pipeline.stages) == 2
    assert pipeline.stages[0] == crew1
    assert pipeline.stages[1] == crew2


@pytest.mark.asyncio
async def test_pipeline_with_empty_input(mock_crew_factory):
    """
    Ensure the pipeline handles an empty input list correctly.
    """
    crew = mock_crew_factory(name="Test Crew")
    pipeline = Pipeline(stages=[crew])

    input_data = []
    pipeline_results = await pipeline.kickoff(input_data)

    assert (
        len(pipeline_results) == 0
    ), "Pipeline should return empty results for empty input"


@pytest.mark.asyncio
async def test_pipeline_process_streams_single_input(mock_crew_factory):
    """
    Test that Pipeline.process_streams() correctly processes a single input
    and returns the expected CrewOutput.
    """
    crew_name = "Test Crew"
    mock_crew = mock_crew_factory(name="Test Crew")
    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key": "value"}]
    pipeline_results = await pipeline.kickoff(input_data)

    mock_crew.kickoff_async.assert_called_once_with(inputs={"key": "value"})

    for pipeline_result in pipeline_results:
        assert isinstance(pipeline_result, PipelineKickoffResult)
        assert pipeline_result.raw == "Test output"
        assert len(pipeline_result.crews_outputs) == 1
        print("pipeline_result.token_usage", pipeline_result.token_usage)
        assert pipeline_result.token_usage == {crew_name: DEFAULT_TOKEN_USAGE}
        assert pipeline_result.trace == [input_data[0], "Test Crew"]


@pytest.mark.asyncio
async def test_pipeline_result_ordering(mock_crew_factory):
    """
    Ensure that results are returned in the same order as the inputs, especially with parallel processing.
    """
    crew1 = mock_crew_factory(name="Crew 1", output_json_dict={"output": "crew1"})
    crew2 = mock_crew_factory(name="Crew 2", output_json_dict={"output": "crew2"})
    crew3 = mock_crew_factory(name="Crew 3", output_json_dict={"output": "crew3"})

    pipeline = Pipeline(
        stages=[crew1, [crew2, crew3]]
    )  # Parallel stage to test ordering

    input_data = [{"id": 1}, {"id": 2}, {"id": 3}]
    pipeline_results = await pipeline.kickoff(input_data)

    assert (
        len(pipeline_results) == 6
    ), "Should have 2 results for each input due to the parallel final stage"

    # Group results by their original input id
    grouped_results = {}
    for result in pipeline_results:
        input_id = result.trace[0]["id"]
        if input_id not in grouped_results:
            grouped_results[input_id] = []
        grouped_results[input_id].append(result)

    # Check that we have the correct number of groups and results per group
    assert len(grouped_results) == 3, "Should have results for each of the 3 inputs"
    for input_id, results in grouped_results.items():
        assert (
            len(results) == 2
        ), f"Each input should have 2 results, but input {input_id} has {len(results)}"

    # Check the ordering and content of the results
    for input_id in range(1, 4):
        group = grouped_results[input_id]
        assert group[0].trace == [
            {"id": input_id},
            "Crew 1",
            "Crew 2",
        ], f"Unexpected trace for first result of input {input_id}"
        assert group[1].trace == [
            {"id": input_id},
            "Crew 1",
            "Crew 3",
        ], f"Unexpected trace for second result of input {input_id}"
        assert (
            group[0].json_dict["output"] == "crew2"
        ), f"Unexpected output for first result of input {input_id}"
        assert (
            group[1].json_dict["output"] == "crew3"
        ), f"Unexpected output for second result of input {input_id}"


class TestPydanticOutput(BaseModel):
    key: str
    value: int


@pytest.mark.asyncio
async def test_pipeline_process_streams_single_input_pydantic_output(mock_crew_factory):
    crew_name = "Test Crew"
    mock_crew = mock_crew_factory(
        name=crew_name,
        output_json_dict=None,
        pydantic_output=TestPydanticOutput(key="test", value=42),
    )
    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key": "value"}]
    pipeline_results = await pipeline.kickoff(input_data)

    assert len(pipeline_results) == 1
    pipeline_result = pipeline_results[0]

    print("pipeline_result.trace", pipeline_result.trace)

    assert isinstance(pipeline_result, PipelineKickoffResult)
    assert pipeline_result.raw == "Test output"
    assert len(pipeline_result.crews_outputs) == 1
    assert pipeline_result.token_usage == {crew_name: DEFAULT_TOKEN_USAGE}
    print("INPUT DATA POST PROCESS", input_data)
    assert pipeline_result.trace == [input_data[0], "Test Crew"]

    assert isinstance(pipeline_result.pydantic, TestPydanticOutput)
    assert pipeline_result.pydantic.key == "test"
    assert pipeline_result.pydantic.value == 42
    assert pipeline_result.json_dict is None


@pytest.mark.asyncio
async def test_pipeline_preserves_original_input(mock_crew_factory):
    crew_name = "Test Crew"
    mock_crew = mock_crew_factory(
        name=crew_name,
        output_json_dict={"new_key": "new_value"},
    )
    pipeline = Pipeline(stages=[mock_crew])

    # Create a deep copy of the input data to ensure we're not comparing references
    original_input_data = [{"key": "value", "nested": {"a": 1}}]
    input_data = json.loads(json.dumps(original_input_data))

    await pipeline.kickoff(input_data)

    # Assert that the original input hasn't been modified
    assert (
        input_data == original_input_data
    ), "The original input data should not be modified"

    # Ensure that even nested structures haven't been modified
    assert (
        input_data[0]["nested"] == original_input_data[0]["nested"]
    ), "Nested structures should not be modified"

    # Verify that adding new keys to the crew output doesn't affect the original input
    assert (
        "new_key" not in input_data[0]
    ), "New keys from crew output should not be added to the original input"


@pytest.mark.asyncio
async def test_pipeline_process_streams_multiple_inputs(mock_crew_factory):
    """
    Test that Pipeline.process_streams() correctly processes multiple inputs
    and returns the expected CrewOutputs.
    """
    mock_crew = mock_crew_factory(name="Test Crew")
    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key1": "value1"}, {"key2": "value2"}]
    pipeline_results = await pipeline.kickoff(input_data)

    assert mock_crew.kickoff_async.call_count == 2
    assert len(pipeline_results) == 2
    for pipeline_result in pipeline_results:
        print("pipeline_result,", pipeline_result)
        assert all(
            isinstance(crew_output, CrewOutput)
            for crew_output in pipeline_result.crews_outputs
        )


@pytest.mark.asyncio
async def test_pipeline_with_parallel_stages(mock_crew_factory):
    """
    Test that Pipeline correctly handles parallel stages.
    """
    crew1 = mock_crew_factory(name="Crew 1")
    crew2 = mock_crew_factory(name="Crew 2")
    crew3 = mock_crew_factory(name="Crew 3")

    pipeline = Pipeline(stages=[crew1, [crew2, crew3]])
    input_data = [{"initial": "data"}]

    pipeline_result = await pipeline.kickoff(input_data)

    crew1.kickoff_async.assert_called_once_with(inputs={"initial": "data"})

    assert len(pipeline_result) == 2
    pipeline_result_1, pipeline_result_2 = pipeline_result

    pipeline_result_1.trace = [
        "Crew 1",
        "Crew 2",
    ]
    pipeline_result_2.trace = [
        "Crew 1",
        "Crew 3",
    ]

    expected_token_usage = {
        "Crew 1": DEFAULT_TOKEN_USAGE,
        "Crew 2": DEFAULT_TOKEN_USAGE,
        "Crew 3": DEFAULT_TOKEN_USAGE,
    }

    assert pipeline_result_1.token_usage == expected_token_usage
    assert pipeline_result_2.token_usage == expected_token_usage


@pytest.mark.asyncio
async def test_pipeline_with_parallel_stages_end_in_single_stage(mock_crew_factory):
    """
    Test that Pipeline correctly handles parallel stages.
    """
    crew1 = mock_crew_factory(name="Crew 1")
    crew2 = mock_crew_factory(name="Crew 2")
    crew3 = mock_crew_factory(name="Crew 3")
    crew4 = mock_crew_factory(name="Crew 4")

    pipeline = Pipeline(stages=[crew1, [crew2, crew3], crew4])
    input_data = [{"initial": "data"}]

    pipeline_result = await pipeline.kickoff(input_data)

    crew1.kickoff_async.assert_called_once_with(inputs={"initial": "data"})

    assert len(pipeline_result) == 1
    pipeline_result_1 = pipeline_result[0]

    pipeline_result_1.trace = [
        input_data[0],
        "Crew 1",
        ["Crew 2", "Crew 3"],
        "Crew 4",
    ]

    expected_token_usage = {
        "Crew 1": DEFAULT_TOKEN_USAGE,
        "Crew 2": DEFAULT_TOKEN_USAGE,
        "Crew 3": DEFAULT_TOKEN_USAGE,
        "Crew 4": DEFAULT_TOKEN_USAGE,
    }

    assert pipeline_result_1.token_usage == expected_token_usage


def test_pipeline_rshift_operator(mock_crew_factory):
    """
    Test that the >> operator correctly creates a Pipeline from Crews and lists of Crews.
    """
    crew1 = mock_crew_factory(name="Crew 1")
    crew2 = mock_crew_factory(name="Crew 2")
    crew3 = mock_crew_factory(name="Crew 3")

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


@pytest.mark.asyncio
async def test_pipeline_parallel_crews_to_parallel_crews(mock_crew_factory):
    """
    Test that feeding parallel crews to parallel crews works correctly.
    """
    crew1 = mock_crew_factory(name="Crew 1", output_json_dict={"output1": "crew1"})
    crew2 = mock_crew_factory(name="Crew 2", output_json_dict={"output2": "crew2"})
    crew3 = mock_crew_factory(name="Crew 3", output_json_dict={"output3": "crew3"})
    crew4 = mock_crew_factory(name="Crew 4", output_json_dict={"output4": "crew4"})

    pipeline = Pipeline(stages=[[crew1, crew2], [crew3, crew4]])

    input_data = [{"input": "test"}]
    pipeline_results = await pipeline.kickoff(input_data)

    assert len(pipeline_results) == 2, "Should have 2 results for final parallel stage"

    pipeline_result_1, pipeline_result_2 = pipeline_results

    # Check the outputs
    assert pipeline_result_1.json_dict == {"output3": "crew3"}
    assert pipeline_result_2.json_dict == {"output4": "crew4"}

    # Check the traces
    expected_traces = [
        [{"input": "test"}, ["Crew 1", "Crew 2"], "Crew 3"],
        [{"input": "test"}, ["Crew 1", "Crew 2"], "Crew 4"],
    ]

    for result, expected_trace in zip(pipeline_results, expected_traces):
        assert result.trace == expected_trace, f"Unexpected trace: {result.trace}"


def test_pipeline_double_nesting_not_allowed(mock_crew_factory):
    """
    Test that double nesting in pipeline stages is not allowed.
    """
    crew1 = mock_crew_factory(name="Crew 1")
    crew2 = mock_crew_factory(name="Crew 2")
    crew3 = mock_crew_factory(name="Crew 3")
    crew4 = mock_crew_factory(name="Crew 4")

    with pytest.raises(ValidationError) as exc_info:
        Pipeline(stages=[crew1, [[crew2, crew3], crew4]])

    error_msg = str(exc_info.value)
    print(f"Full error message: {error_msg}")  # For debugging
    assert (
        "Double nesting is not allowed in pipeline stages" in error_msg
    ), f"Unexpected error message: {error_msg}"


def test_pipeline_invalid_crew(mock_crew_factory):
    """
    Test that non-Crew objects are not allowed in pipeline stages.
    """
    crew1 = mock_crew_factory(name="Crew 1")
    not_a_crew = "This is not a crew"

    with pytest.raises(ValidationError) as exc_info:
        Pipeline(stages=[crew1, not_a_crew])

    error_msg = str(exc_info.value)
    print(f"Full error message: {error_msg}")  # For debugging
    assert (
        "Expected Crew instance or list of Crews, got <class 'str'>" in error_msg
    ), f"Unexpected error message: {error_msg}"


"""
TODO: Figure out what is the proper output for a pipeline with multiple stages

Options:
- Should the final output only include the last stage's output?
- Should the final output include the accumulation of previous stages' outputs?
"""


@pytest.mark.asyncio
async def test_pipeline_data_accumulation(mock_crew_factory):
    crew1 = mock_crew_factory(name="Crew 1", output_json_dict={"key1": "value1"})
    crew2 = mock_crew_factory(name="Crew 2", output_json_dict={"key2": "value2"})

    pipeline = Pipeline(stages=[crew1, crew2])
    input_data = [{"initial": "data"}]
    results = await pipeline.kickoff(input_data)

    # Check that crew1 was called with only the initial input
    crew1.kickoff_async.assert_called_once_with(inputs={"initial": "data"})

    # Check that crew2 was called with the combined input from the initial data and crew1's output
    crew2.kickoff_async.assert_called_once_with(
        inputs={"initial": "data", "key1": "value1"}
    )

    # Check the final output
    assert len(results) == 1
    final_result = results[0]
    assert final_result.json_dict == {"key2": "value2"}

    # Check that the trace includes all stages
    assert final_result.trace == [{"initial": "data"}, "Crew 1", "Crew 2"]

    # Check that crews_outputs contain the correct information
    assert len(final_result.crews_outputs) == 2
    assert final_result.crews_outputs[0].json_dict == {"key1": "value1"}
    assert final_result.crews_outputs[1].json_dict == {"key2": "value2"}
