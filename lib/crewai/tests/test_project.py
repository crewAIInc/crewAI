from typing import Any, ClassVar
from unittest.mock import Mock, patch

import pytest
from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.llm import LLM
from crewai.project import (
    CrewBase,
    after_kickoff,
    agent,
    before_kickoff,
    crew,
    llm,
    task,
)
from crewai.task import Task
from crewai.tools import tool


class SimpleCrew:
    @agent
    def simple_agent(self):
        return Agent(
            role="Simple Agent", goal="Simple Goal", backstory="Simple Backstory"
        )

    @task
    def simple_task(self):
        return Task(description="Simple Description", expected_output="Simple Output")

    @task
    def custom_named_task(self):
        return Task(
            description="Simple Description",
            expected_output="Simple Output",
            name="Custom",
        )


@CrewBase
class InternalCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    @llm
    def local_llm(self):
        return LLM(
            model="openai/model_name",
            api_key="None",
            base_url="http://xxx.xxx.xxx.xxx:8000/v1",
        )

    @agent
    def researcher(self):
        return Agent(config=self.agents_config["researcher"])  # type: ignore[index]

    @agent
    def reporting_analyst(self):
        return Agent(config=self.agents_config["reporting_analyst"])  # type: ignore[index]

    @task
    def research_task(self):
        return Task(config=self.tasks_config["research_task"])  # type: ignore[index]

    @task
    def reporting_task(self):
        return Task(config=self.tasks_config["reporting_task"])  # type: ignore[index]

    @before_kickoff
    def modify_inputs(self, inputs):
        if inputs:
            inputs["topic"] = "Bicycles"
        return inputs

    @after_kickoff
    def modify_outputs(self, outputs):
        outputs.raw = outputs.raw + " post processed"
        return outputs

    @crew
    def crew(self):
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class InternalCrewWithMCP(InternalCrew):
    mcp_server_params: ClassVar[dict[str, Any]] = {"host": "localhost", "port": 8000}
    mcp_connect_timeout = 120

    @agent
    def reporting_analyst(self):
        return Agent(
            config=self.agents_config["reporting_analyst"], tools=self.get_mcp_tools()
        )  # type: ignore[index]

    @agent
    def researcher(self):
        return Agent(
            config=self.agents_config["researcher"],
            tools=self.get_mcp_tools("simple_tool"),
        )  # type: ignore[index]


def test_agent_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_agent()
    second_call_result = crew.simple_agent()

    assert first_call_result is second_call_result, (
        "Agent memoization is not working as expected"
    )


def test_task_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_task()
    second_call_result = crew.simple_task()

    assert first_call_result is second_call_result, (
        "Task memoization is not working as expected"
    )


def test_crew_memoization():
    crew = InternalCrew()
    first_call_result = crew.crew()
    second_call_result = crew.crew()

    assert first_call_result is second_call_result, (
        "Crew references should point to the same object"
    )


def test_task_name():
    simple_task = SimpleCrew().simple_task()
    assert simple_task.name == "simple_task", (
        "Task name is not inferred from function name as expected"
    )

    custom_named_task = SimpleCrew().custom_named_task()
    assert custom_named_task.name == "Custom", (
        "Custom task name is not being set as expected"
    )


def test_agent_function_calling_llm():
    crew = InternalCrew()
    llm = crew.local_llm()
    obj_llm_agent = crew.researcher()
    assert obj_llm_agent.function_calling_llm is llm, (
        "agent's function_calling_llm is incorrect"
    )

    str_llm_agent = crew.reporting_analyst()
    assert str_llm_agent.function_calling_llm.model == "online_llm", (
        "agent's function_calling_llm is incorrect"
    )


def test_task_guardrail():
    crew = InternalCrew()
    research_task = crew.research_task()
    assert research_task.guardrail == "ensure each bullet contains its source"

    reporting_task = crew.reporting_task()
    assert reporting_task.guardrail is None


@pytest.mark.vcr()
def test_before_kickoff_modification():
    crew = InternalCrew()
    inputs = {"topic": "LLMs"}
    result = crew.crew().kickoff(inputs=inputs)
    assert "bicycles" in result.raw, "Before kickoff function did not modify inputs"


@pytest.mark.vcr()
def test_after_kickoff_modification():
    crew = InternalCrew()
    # Assuming the crew execution returns a dict
    result = crew.crew().kickoff({"topic": "LLMs"})

    assert "post processed" in result.raw, (
        "After kickoff function did not modify outputs"
    )


@pytest.mark.vcr()
def test_before_kickoff_with_none_input():
    crew = InternalCrew()
    crew.crew().kickoff(None)
    # Test should pass without raising exceptions


@pytest.mark.vcr()
def test_multiple_before_after_kickoff():
    @CrewBase
    class MultipleHooksCrew:
        agents: list[BaseAgent]
        tasks: list[Task]

        agents_config = "config/agents.yaml"
        tasks_config = "config/tasks.yaml"

        @agent
        def researcher(self):
            return Agent(config=self.agents_config["researcher"])  # type: ignore[index]

        @agent
        def reporting_analyst(self):
            return Agent(config=self.agents_config["reporting_analyst"])  # type: ignore[index]

        @task
        def research_task(self):
            return Task(config=self.tasks_config["research_task"])  # type: ignore[index]

        @task
        def reporting_task(self):
            return Task(config=self.tasks_config["reporting_task"])  # type: ignore[index]

        @before_kickoff
        def first_before(self, inputs):
            inputs["topic"] = "Bicycles"
            return inputs

        @before_kickoff
        def second_before(self, inputs):
            inputs["topic"] = "plants"
            return inputs

        @after_kickoff
        def first_after(self, outputs):
            outputs.raw = outputs.raw + " processed first"
            return outputs

        @after_kickoff
        def second_after(self, outputs):
            outputs.raw = outputs.raw + " processed second"
            return outputs

        @crew
        def crew(self):
            return Crew(agents=self.agents, tasks=self.tasks, verbose=True)

    crew = MultipleHooksCrew()
    result = crew.crew().kickoff({"topic": "LLMs"})

    assert "plants" in result.raw, "First before_kickoff not executed"
    assert "processed first" in result.raw, "First after_kickoff not executed"
    assert "processed second" in result.raw, "Second after_kickoff not executed"


def test_crew_name():
    crew = InternalCrew()
    assert crew._crew_name == "InternalCrew"


@tool
def simple_tool():
    """Return 'Hi!'"""
    return "Hi!"


@tool
def another_simple_tool():
    """Return 'Hi!'"""
    return "Hi!"


class TestAsyncDecoratorSupport:
    """Tests for async method support in @agent, @task decorators."""

    def test_async_agent_memoization(self):
        """Async agent methods should be properly memoized."""

        class AsyncAgentCrew:
            call_count = 0

            @agent
            async def async_agent(self):
                AsyncAgentCrew.call_count += 1
                return Agent(
                    role="Async Agent", goal="Async Goal", backstory="Async Backstory"
                )

        crew = AsyncAgentCrew()
        first_call = crew.async_agent()
        second_call = crew.async_agent()

        assert first_call is second_call, "Async agent memoization failed"
        assert AsyncAgentCrew.call_count == 1, "Async agent called more than once"

    def test_async_task_memoization(self):
        """Async task methods should be properly memoized."""

        class AsyncTaskCrew:
            call_count = 0

            @task
            async def async_task(self):
                AsyncTaskCrew.call_count += 1
                return Task(
                    description="Async Description", expected_output="Async Output"
                )

        crew = AsyncTaskCrew()
        first_call = crew.async_task()
        second_call = crew.async_task()

        assert first_call is second_call, "Async task memoization failed"
        assert AsyncTaskCrew.call_count == 1, "Async task called more than once"

    def test_async_task_name_inference(self):
        """Async task should have name inferred from method name."""

        class AsyncTaskNameCrew:
            @task
            async def my_async_task(self):
                return Task(
                    description="Async Description", expected_output="Async Output"
                )

        crew = AsyncTaskNameCrew()
        task_instance = crew.my_async_task()

        assert task_instance.name == "my_async_task", (
            "Async task name not inferred correctly"
        )

    def test_async_agent_returns_agent_not_coroutine(self):
        """Async agent decorator should return Agent, not coroutine."""

        class AsyncAgentTypeCrew:
            @agent
            async def typed_async_agent(self):
                return Agent(
                    role="Typed Agent", goal="Typed Goal", backstory="Typed Backstory"
                )

        crew = AsyncAgentTypeCrew()
        result = crew.typed_async_agent()

        assert isinstance(result, Agent), (
            f"Expected Agent, got {type(result).__name__}"
        )

    def test_async_task_returns_task_not_coroutine(self):
        """Async task decorator should return Task, not coroutine."""

        class AsyncTaskTypeCrew:
            @task
            async def typed_async_task(self):
                return Task(
                    description="Typed Description", expected_output="Typed Output"
                )

        crew = AsyncTaskTypeCrew()
        result = crew.typed_async_task()

        assert isinstance(result, Task), f"Expected Task, got {type(result).__name__}"


def test_internal_crew_with_mcp():
    from crewai_tools.adapters.tool_collection import ToolCollection

    mock_adapter = Mock()
    mock_adapter.tools = ToolCollection([simple_tool, another_simple_tool])

    with (
        patch("crewai_tools.MCPServerAdapter", return_value=mock_adapter) as adapter_mock,
        patch("crewai.llm.LLM.__new__", return_value=Mock()),
    ):
        crew = InternalCrewWithMCP()
        assert crew.reporting_analyst().tools == [simple_tool, another_simple_tool]
        assert crew.researcher().tools == [simple_tool]

    adapter_mock.assert_called_once_with(
        {"host": "localhost", "port": 8000}, connect_timeout=120
    )
