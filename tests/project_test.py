from typing import List
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

    agents: List[BaseAgent]
    tasks: List[Task]

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
    mcp_server_params = {"host": "localhost", "port": 8000}

    @agent
    def reporting_analyst(self):
        return Agent(config=self.agents_config["reporting_analyst"], tools=self.get_mcp_tools())  # type: ignore[index]

    @agent
    def researcher(self):
        return Agent(config=self.agents_config["researcher"], tools=self.get_mcp_tools("simple_tool"))  # type: ignore[index]

def test_agent_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_agent()
    second_call_result = crew.simple_agent()

    assert (
        first_call_result is second_call_result
    ), "Agent memoization is not working as expected"


def test_task_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_task()
    second_call_result = crew.simple_task()

    assert (
        first_call_result is second_call_result
    ), "Task memoization is not working as expected"


def test_crew_memoization():
    crew = InternalCrew()
    first_call_result = crew.crew()
    second_call_result = crew.crew()

    assert (
        first_call_result is second_call_result
    ), "Crew references should point to the same object"


def test_task_name():
    simple_task = SimpleCrew().simple_task()
    assert (
        simple_task.name == "simple_task"
    ), "Task name is not inferred from function name as expected"

    custom_named_task = SimpleCrew().custom_named_task()
    assert (
        custom_named_task.name == "Custom"
    ), "Custom task name is not being set as expected"


def test_agent_function_calling_llm():
    crew = InternalCrew()
    llm = crew.local_llm()
    obj_llm_agent = crew.researcher()
    assert (
        obj_llm_agent.function_calling_llm is llm
    ), "agent's function_calling_llm is incorrect"

    str_llm_agent = crew.reporting_analyst()
    assert (
        str_llm_agent.function_calling_llm.model == "online_llm"
    ), "agent's function_calling_llm is incorrect"


def test_task_guardrail():
    crew = InternalCrew()
    research_task = crew.research_task()
    assert research_task.guardrail == "ensure each bullet contains its source"

    reporting_task = crew.reporting_task()
    assert reporting_task.guardrail is None


@pytest.mark.vcr(filter_headers=["authorization"])
def test_before_kickoff_modification():
    crew = InternalCrew()
    inputs = {"topic": "LLMs"}
    result = crew.crew().kickoff(inputs=inputs)
    assert "bicycles" in result.raw, "Before kickoff function did not modify inputs"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_after_kickoff_modification():
    crew = InternalCrew()
    # Assuming the crew execution returns a dict
    result = crew.crew().kickoff({"topic": "LLMs"})

    assert (
        "post processed" in result.raw
    ), "After kickoff function did not modify outputs"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_before_kickoff_with_none_input():
    crew = InternalCrew()
    crew.crew().kickoff(None)
    # Test should pass without raising exceptions


@pytest.mark.vcr(filter_headers=["authorization"])
def test_multiple_before_after_kickoff():
    @CrewBase
    class MultipleHooksCrew:
        agents: List[BaseAgent]
        tasks: List[Task]

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


def test_internal_crew_with_mcp():
    from crewai_tools import MCPServerAdapter
    from crewai_tools.adapters.mcp_adapter import ToolCollection
    mock = Mock(spec=MCPServerAdapter)
    mock.tools = ToolCollection([simple_tool, another_simple_tool])
    with patch("crewai_tools.MCPServerAdapter", return_value=mock) as adapter_mock:
        crew = InternalCrewWithMCP()
        assert crew.reporting_analyst().tools == [simple_tool, another_simple_tool]
        assert crew.researcher().tools == [simple_tool]

    adapter_mock.assert_called_once_with({"host": "localhost", "port": 8000})