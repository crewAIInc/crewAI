import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.project import CrewBase, after_kickoff, agent, before_kickoff, crew, task
from crewai.task import Task


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

    @agent
    def researcher(self):
        return Agent(config=self.agents_config["researcher"])

    @agent
    def reporting_analyst(self):
        return Agent(config=self.agents_config["reporting_analyst"])

    @task
    def research_task(self):
        return Task(config=self.tasks_config["research_task"])

    @task
    def reporting_task(self):
        return Task(config=self.tasks_config["reporting_task"])

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
        agents_config = "config/agents.yaml"
        tasks_config = "config/tasks.yaml"

        @agent
        def researcher(self):
            return Agent(config=self.agents_config["researcher"])

        @agent
        def reporting_analyst(self):
            return Agent(config=self.agents_config["reporting_analyst"])

        @task
        def research_task(self):
            return Task(config=self.tasks_config["research_task"])

        @task
        def reporting_task(self):
            return Task(config=self.tasks_config["reporting_task"])

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
