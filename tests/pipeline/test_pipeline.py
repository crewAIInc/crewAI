import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.pipeline.pipeline import Pipeline
from crewai.pipeline.pipeline_kickoff_result import PipelineKickoffResult
from crewai.process import Process
from crewai.routers.router import Route, Router
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
        MockCrewClass = type("MockCrew", (MagicMock, Crew), {})

        class MockCrew(MockCrewClass):
            def __deepcopy__(self):
                result = MockCrewClass()
                result.kickoff_async = self.kickoff_async
                result.name = self.name
                return result

            def copy(
                self,
            ):
                return self

        crew = MockCrew()
        crew.name = name

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

        async def kickoff_async(inputs=None):
            return crew_output

        # Create an AsyncMock for kickoff_async
        crew.kickoff_async = AsyncMock(side_effect=kickoff_async)

        # Mock the synchronous kickoff method
        crew.kickoff = MagicMock(return_value=crew_output)

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


@pytest.fixture
def mock_router_factory(mock_crew_factory):
    def _create_mock_router():
        crew1 = mock_crew_factory(name="Crew 1", output_json_dict={"output": "crew1"})
        crew2 = mock_crew_factory(name="Crew 2", output_json_dict={"output": "crew2"})
        crew3 = mock_crew_factory(name="Crew 3", output_json_dict={"output": "crew3"})

        MockRouterClass = type("MockRouter", (MagicMock, Router), {})

        class MockRouter(MockRouterClass):
            def __deepcopy__(self, memo):
                result = MockRouterClass()
                result.route = self.route
                return result

        mock_router = MockRouter()
        mock_router.route = MagicMock(
            side_effect=lambda x: (
                (
                    Pipeline(stages=[crew1])
                    if x.get("score", 0) > 80
                    else (
                        Pipeline(stages=[crew2])
                        if x.get("score", 0) > 50
                        else Pipeline(stages=[crew3])
                    )
                ),
                (
                    "route1"
                    if x.get("score", 0) > 80
                    else "route2"
                    if x.get("score", 0) > 50
                    else "default"
                ),
            )
        )

        return mock_router

    return _create_mock_router


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


agent = Agent(
    role="Test Role",
    goal="Test Goal",
    backstory="Test Backstory",
    allow_delegation=False,
    verbose=False,
)
task = Task(
    description="Return: Test output",
    expected_output="Test output",
    agent=agent,
    async_execution=False,
    context=None,
)


@pytest.mark.asyncio
async def test_pipeline_process_streams_single_input():
    """
    Test that Pipeline.process_streams() correctly processes a single input
    and returns the expected CrewOutput.
    """
    crew_name = "Test Crew"
    mock_crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
    )
    mock_crew.name = crew_name
    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key": "value"}]
    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        task_output = TaskOutput(
            description="Test task", raw="Task output", agent="Test Agent"
        )
        mock_kickoff.return_value = CrewOutput(
            raw="Test output",
            tasks_output=[task_output],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict=None,
            pydantic=None,
        )
        pipeline_results = await pipeline.kickoff(input_data)
        mock_crew.kickoff_async.assert_called_once_with(inputs={"key": "value"})

        for pipeline_result in pipeline_results:
            assert isinstance(pipeline_result, PipelineKickoffResult)
            assert pipeline_result.raw == "Test output"
            assert len(pipeline_result.crews_outputs) == 1
            assert pipeline_result.token_usage == {crew_name: DEFAULT_TOKEN_USAGE}
            assert pipeline_result.trace == [input_data[0], "Test Crew"]


@pytest.mark.asyncio
async def test_pipeline_result_ordering():
    """
    Ensure that results are returned in the same order as the inputs, especially with parallel processing.
    """
    crew1 = Crew(
        name="Crew 1",
        agents=[agent],
        tasks=[task],
    )
    crew2 = Crew(
        name="Crew 2",
        agents=[agent],
        tasks=[task],
    )
    crew3 = Crew(
        name="Crew 3",
        agents=[agent],
        tasks=[task],
    )

    pipeline = Pipeline(
        stages=[crew1, [crew2, crew3]]
    )  # Parallel stage to test ordering
    input_data = [{"id": 1}, {"id": 2}, {"id": 3}]

    def create_crew_output(crew_name):
        return CrewOutput(
            raw=f"Test output from {crew_name}",
            tasks_output=[
                TaskOutput(
                    description="Test task",
                    raw=f"Task output from {crew_name}",
                    agent="Test Agent",
                )
            ],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict={"output": crew_name.lower().replace(" ", "")},
            pydantic=None,
        )

    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.side_effect = [
            create_crew_output("Crew 1"),
            create_crew_output("Crew 2"),
            create_crew_output("Crew 3"),
        ] * 3
        pipeline_results = await pipeline.kickoff(input_data)
        mock_kickoff.call_count = 3

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


class TestPydanticOutput(BaseModel):
    key: str
    value: int


@pytest.mark.asyncio
@pytest.mark.vcr(filter_headers=["authorization"])
async def test_pipeline_process_streams_single_input_pydantic_output():
    crew_name = "Test Crew"
    task = Task(
        description="Return: Key:value",
        expected_output="Key:Value",
        agent=agent,
        async_execution=False,
        context=None,
        output_pydantic=TestPydanticOutput,
    )
    mock_crew = Crew(
        name=crew_name,
        agents=[agent],
        tasks=[task],
    )

    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key": "value"}]
    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_crew_output = CrewOutput(
            raw="Test output",
            tasks_output=[
                TaskOutput(
                    description="Return: Key:value", raw="Key:Value", agent="Test Agent"
                )
            ],
            token_usage=UsageMetrics(
                total_tokens=171,
                prompt_tokens=154,
                completion_tokens=17,
                successful_requests=1,
            ),
            pydantic=TestPydanticOutput(key="test", value=42),
        )
        mock_kickoff.return_value = mock_crew_output
        pipeline_results = await pipeline.kickoff(input_data)

    assert len(pipeline_results) == 1
    pipeline_result = pipeline_results[0]

    assert isinstance(pipeline_result, PipelineKickoffResult)
    assert pipeline_result.raw == "Test output"
    assert len(pipeline_result.crews_outputs) == 1
    assert pipeline_result.token_usage == {
        crew_name: UsageMetrics(
            total_tokens=171,
            prompt_tokens=154,
            completion_tokens=17,
            successful_requests=1,
        )
    }

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
async def test_pipeline_process_streams_multiple_inputs():
    """
    Test that Pipeline.process_streams() correctly processes multiple inputs
    and returns the expected CrewOutputs.
    """
    mock_crew = Crew(name="Test Crew", tasks=[task], agents=[agent])
    pipeline = Pipeline(stages=[mock_crew])
    input_data = [{"key1": "value1"}, {"key2": "value2"}]

    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.return_value = CrewOutput(
            raw="Test output",
            tasks_output=[
                TaskOutput(
                    description="Test task", raw="Task output", agent="Test Agent"
                )
            ],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict=None,
            pydantic=None,
        )
        pipeline_results = await pipeline.kickoff(input_data)
        assert mock_kickoff.call_count == 2
        assert len(pipeline_results) == 2

    for pipeline_result in pipeline_results:
        assert all(
            isinstance(crew_output, CrewOutput)
            for crew_output in pipeline_result.crews_outputs
        )


@pytest.mark.asyncio
async def test_pipeline_with_parallel_stages():
    """
    Test that Pipeline correctly handles parallel stages.
    """
    crew1 = Crew(name="Crew 1", tasks=[task], agents=[agent])
    crew2 = Crew(name="Crew 2", tasks=[task], agents=[agent])
    crew3 = Crew(name="Crew 3", tasks=[task], agents=[agent])

    pipeline = Pipeline(stages=[crew1, [crew2, crew3]])
    input_data = [{"initial": "data"}]

    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.return_value = CrewOutput(
            raw="Test output",
            tasks_output=[
                TaskOutput(
                    description="Test task", raw="Task output", agent="Test Agent"
                )
            ],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict=None,
            pydantic=None,
        )
        pipeline_result = await pipeline.kickoff(input_data)
        mock_kickoff.assert_called_with(inputs={"initial": "data"})

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
    assert len(pipeline.stages) == 2
    assert pipeline.stages[1] == [crew2, crew3]

    # Test error case: trying to shift with non-Crew object
    with pytest.raises(TypeError):
        pipeline >> "not a crew"


@pytest.mark.asyncio
@pytest.mark.vcr(filter_headers=["authorization"])
async def test_pipeline_parallel_crews_to_parallel_crews():
    """
    Test that feeding parallel crews to parallel crews works correctly.
    """
    crew1 = Crew(name="Crew 1", tasks=[task], agents=[agent])
    crew2 = Crew(name="Crew 2", tasks=[task], agents=[agent])
    crew3 = Crew(name="Crew 3", tasks=[task], agents=[agent])
    crew4 = Crew(name="Crew 4", tasks=[task], agents=[agent])
    #  output_json_dict={"output1": "crew1"}
    pipeline = Pipeline(stages=[[crew1, crew2], [crew3, crew4]])

    input_data = [{"input": "test"}]

    def create_crew_output(crew_name):
        return CrewOutput(
            raw=f"Test output from {crew_name}",
            tasks_output=[
                TaskOutput(
                    description="Test task",
                    raw=f"Task output from {crew_name}",
                    agent="Test Agent",
                )
            ],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict={"output": crew_name.lower().replace(" ", "")},
            pydantic=None,
        )

    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.side_effect = [
            create_crew_output(crew_name)
            for crew_name in ["Crew 1", "Crew 2", "Crew 3", "Crew 4"]
        ]
        pipeline_results = await pipeline.kickoff(input_data)

    assert len(pipeline_results) == 2, "Should have 2 results for final parallel stage"

    pipeline_result_1, pipeline_result_2 = pipeline_results

    # Check the outputs
    assert pipeline_result_1.json_dict == {"output": "crew3"}
    assert pipeline_result_2.json_dict == {"output": "crew4"}

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
    assert (
        "Expected Crew instance, Router instance, or list of Crews, got <class 'str'>"
        in error_msg
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


@pytest.mark.asyncio
@pytest.mark.vcr(filter_headers=["authorization"])
async def test_pipeline_with_router():
    crew1 = Crew(name="Crew 1", tasks=[task], agents=[agent])
    crew2 = Crew(name="Crew 2", tasks=[task], agents=[agent])
    crew3 = Crew(name="Crew 3", tasks=[task], agents=[agent])
    routes = {
        "route1": Route(
            condition=lambda x: x.get("score", 0) > 80,
            pipeline=Pipeline(stages=[crew1]),
        ),
        "route2": Route(
            condition=lambda x: 50 < x.get("score", 0) <= 80,
            pipeline=Pipeline(stages=[crew2]),
        ),
    }
    router = Router(
        routes=routes,
        default=Pipeline(stages=[crew3]),
    )
    # Test high score route
    pipeline = Pipeline(stages=[router])
    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.return_value = CrewOutput(
            raw="Test output from Crew 1",
            tasks_output=[],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict={"output": "crew1"},
            pydantic=None,
        )
        result_high = await pipeline.kickoff([{"score": 90}])

        assert len(result_high) == 1
        assert result_high[0].json_dict is not None
        assert result_high[0].json_dict["output"] == "crew1"
        assert result_high[0].trace == [
            {"score": 90},
            {"route_taken": "route1"},
            "Crew 1",
        ]
    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.return_value = CrewOutput(
            raw="Test output from Crew 2",
            tasks_output=[],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict={"output": "crew2"},
            pydantic=None,
        )
        # Test medium score route
        pipeline = Pipeline(stages=[router])
        result_medium = await pipeline.kickoff([{"score": 60}])
        assert len(result_medium) == 1
        assert result_medium[0].json_dict is not None
        assert result_medium[0].json_dict["output"] == "crew2"
        assert result_medium[0].trace == [
            {"score": 60},
            {"route_taken": "route2"},
            "Crew 2",
        ]

    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.return_value = CrewOutput(
            raw="Test output from Crew 3",
            tasks_output=[],
            token_usage=DEFAULT_TOKEN_USAGE,
            json_dict={"output": "crew3"},
            pydantic=None,
        )
        # Test low score route
        pipeline = Pipeline(stages=[router])
        result_low = await pipeline.kickoff([{"score": 30}])
        assert len(result_low) == 1
        assert result_low[0].json_dict is not None
        assert result_low[0].json_dict["output"] == "crew3"
        assert result_low[0].trace == [
            {"score": 30},
            {"route_taken": "default"},
            "Crew 3",
        ]


@pytest.mark.asyncio
@pytest.mark.vcr(filter_headers=["authorization"])
async def test_router_with_multiple_inputs():
    crew1 = Crew(name="Crew 1", tasks=[task], agents=[agent])
    crew2 = Crew(name="Crew 2", tasks=[task], agents=[agent])
    crew3 = Crew(name="Crew 3", tasks=[task], agents=[agent])
    router = Router(
        routes={
            "route1": Route(
                condition=lambda x: x.get("score", 0) > 80,
                pipeline=Pipeline(stages=[crew1]),
            ),
            "route2": Route(
                condition=lambda x: 50 < x.get("score", 0) <= 80,
                pipeline=Pipeline(stages=[crew2]),
            ),
        },
        default=Pipeline(stages=[crew3]),
    )
    pipeline = Pipeline(stages=[router])

    inputs = [{"score": 90}, {"score": 60}, {"score": 30}]

    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.side_effect = [
            CrewOutput(
                raw="Test output from Crew 1",
                tasks_output=[],
                token_usage=DEFAULT_TOKEN_USAGE,
                json_dict={"output": "crew1"},
                pydantic=None,
            ),
            CrewOutput(
                raw="Test output from Crew 2",
                tasks_output=[],
                token_usage=DEFAULT_TOKEN_USAGE,
                json_dict={"output": "crew2"},
                pydantic=None,
            ),
            CrewOutput(
                raw="Test output from Crew 3",
                tasks_output=[],
                token_usage=DEFAULT_TOKEN_USAGE,
                json_dict={"output": "crew3"},
                pydantic=None,
            ),
        ]
        results = await pipeline.kickoff(inputs)

    assert len(results) == 3
    assert results[0].json_dict is not None
    assert results[0].json_dict["output"] == "crew1"
    assert results[1].json_dict is not None
    assert results[1].json_dict["output"] == "crew2"
    assert results[2].json_dict is not None
    assert results[2].json_dict["output"] == "crew3"

    assert results[0].trace[1]["route_taken"] == "route1"
    assert results[1].trace[1]["route_taken"] == "route2"
    assert results[2].trace[1]["route_taken"] == "default"


@pytest.mark.asyncio
@pytest.mark.vcr(filter_headers=["authorization"])
async def test_pipeline_with_multiple_routers():
    crew1 = Crew(name="Crew 1", tasks=[task], agents=[agent])
    crew2 = Crew(name="Crew 2", tasks=[task], agents=[agent])
    router1 = Router(
        routes={
            "route1": Route(
                condition=lambda x: x.get("score", 0) > 80,
                pipeline=Pipeline(stages=[crew1]),
            ),
        },
        default=Pipeline(stages=[crew2]),
    )
    router2 = Router(
        routes={
            "route2": Route(
                condition=lambda x: 50 < x.get("score", 0) <= 80,
                pipeline=Pipeline(stages=[crew2]),
            ),
        },
        default=Pipeline(stages=[crew2]),
    )
    final_crew = Crew(name="Final Crew", tasks=[task], agents=[agent])

    pipeline = Pipeline(stages=[router1, router2, final_crew])

    with patch.object(Crew, "kickoff_async") as mock_kickoff:
        mock_kickoff.side_effect = [
            CrewOutput(
                raw="Test output from Crew 1",
                tasks_output=[],
                token_usage=DEFAULT_TOKEN_USAGE,
                json_dict={"output": "crew1"},
                pydantic=None,
            ),
            CrewOutput(
                raw="Test output from Crew 2",
                tasks_output=[],
                token_usage=DEFAULT_TOKEN_USAGE,
                json_dict={"output": "crew2"},
                pydantic=None,
            ),
            CrewOutput(
                raw="Test output from Final Crew",
                tasks_output=[],
                token_usage=DEFAULT_TOKEN_USAGE,
                json_dict={"output": "final"},
                pydantic=None,
            ),
        ]
        result = await pipeline.kickoff([{"score": 75}])

    assert len(result) == 1
    assert result[0].json_dict is not None
    assert result[0].json_dict["output"] == "final"
    assert (
        len(result[0].trace) == 6
    )  # Input, Router1, Crew2, Router2, Crew2, Final Crew
    assert result[0].trace[1]["route_taken"] == "default"
    assert result[0].trace[3]["route_taken"] == "route2"


@pytest.mark.asyncio
async def test_router_default_route(mock_crew_factory):
    default_crew = mock_crew_factory(
        name="Default Crew", output_json_dict={"output": "default"}
    )
    router = Router(
        routes={
            "route1": Route(
                condition=lambda x: False,
                pipeline=Pipeline(stages=[mock_crew_factory(name="Never Used")]),
            ),
        },
        default=Pipeline(stages=[default_crew]),
    )

    pipeline = Pipeline(stages=[router])
    result = await pipeline.kickoff([{"score": 100}])

    assert len(result) == 1
    assert result[0].json_dict is not None
    assert result[0].json_dict["output"] == "default"
    assert result[0].trace[1]["route_taken"] == "default"


@pytest.mark.asyncio
@pytest.mark.vcr(filter_headers=["authorization"])
async def test_router_with_empty_input():
    crew1 = Crew(name="Crew 1", tasks=[task], agents=[agent])
    crew2 = Crew(name="Crew 2", tasks=[task], agents=[agent])
    crew3 = Crew(name="Crew 3", tasks=[task], agents=[agent])
    router = Router(
        routes={
            "route1": Route(
                condition=lambda x: x.get("score", 0) > 80,
                pipeline=Pipeline(stages=[crew1]),
            ),
            "route2": Route(
                condition=lambda x: 50 < x.get("score", 0) <= 80,
                pipeline=Pipeline(stages=[crew2]),
            ),
        },
        default=Pipeline(stages=[crew3]),
    )
    pipeline = Pipeline(stages=[router])

    result = await pipeline.kickoff([{}])

    assert len(result) == 1
    assert result[0].trace[1]["route_taken"] == "default"
