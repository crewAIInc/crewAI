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
class TestCrew:
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
    crew = TestCrew()
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
    crew = TestCrew()
    inputs = {"topic": "LLMs"}
    result = crew.crew().kickoff(inputs=inputs)
    assert "bicycles" in result.raw, "Before kickoff function did not modify inputs"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_after_kickoff_modification():
    crew = TestCrew()
    # Assuming the crew execution returns a dict
    result = crew.crew().kickoff({"topic": "LLMs"})

    assert (
        "post processed" in result.raw
    ), "After kickoff function did not modify outputs"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_before_kickoff_with_none_input():
    crew = TestCrew()
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_multiple_yaml_configs():
    @CrewBase
    class MultiConfigCrew:
        agents_config = ["config/multi/agents1.yaml", "config/multi/agents2.yaml"]
        tasks_config = ["config/multi/tasks1.yaml", "config/multi/tasks2.yaml"]

        @agent
        def test_agent1(self):
            return Agent(config=self.agents_config["test_agent1"])

        @agent
        def test_agent2(self):
            return Agent(config=self.agents_config["test_agent2"])

        @task
        def test_task1(self):
            task_config = self.tasks_config["test_task1"].copy()
            if isinstance(task_config.get("agent"), str):
                agent_name = task_config.pop("agent")
                if hasattr(self, agent_name):
                    task_config["agent"] = getattr(self, agent_name)()
            return Task(config=task_config)

        @task
        def test_task2(self):
            task_config = self.tasks_config["test_task2"].copy()
            if isinstance(task_config.get("agent"), str):
                agent_name = task_config.pop("agent")
                if hasattr(self, agent_name):
                    task_config["agent"] = getattr(self, agent_name)()
            return Task(config=task_config)

        @crew
        def crew(self):
            return Crew(agents=self.agents, tasks=self.tasks, verbose=True)

    crew = MultiConfigCrew()
    
    assert "test_agent1" in crew.agents_config
    assert "test_agent2" in crew.agents_config
    
    assert crew.agents_config["test_agent1"]["role"] == "Updated Test Agent 1"
    assert crew.agents_config["test_agent1"]["goal"] == "Updated Test Goal 1"
    assert crew.agents_config["test_agent1"]["backstory"] == "Test Backstory 1"
    assert crew.agents_config["test_agent1"]["verbose"] is True
    
    assert "test_task1" in crew.tasks_config
    assert "test_task2" in crew.tasks_config
    
    assert crew.tasks_config["test_task1"]["description"] == "Updated Test Description 1"
    assert crew.tasks_config["test_task1"]["expected_output"] == "Test Output 1"
    assert crew.tasks_config["test_task1"]["agent"].role == "Updated Test Agent 1"

    agent1 = crew.test_agent1()
    agent2 = crew.test_agent2()
    task1 = crew.test_task1()
    task2 = crew.test_task2()
    
    assert agent1.role == "Updated Test Agent 1"
    assert agent2.role == "Test Agent 2"
    assert task1.description == "Updated Test Description 1"
    assert task2.description == "Test Description 2"
