from unittest.mock import patch

import pytest

from crewai.agent import Agent
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.planning_handler import (
    CrewPlanner,
    PlannerTaskPydanticOutput,
    PlanPerTask,
)


class TestCrewPlanner:
    @pytest.fixture
    def crew_planner(self):
        tasks = [
            Task(
                description="Task 1",
                expected_output="Output 1",
                agent=Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1"),
            ),
            Task(
                description="Task 2",
                expected_output="Output 2",
                agent=Agent(role="Agent 2", goal="Goal 2", backstory="Backstory 2"),
            ),
            Task(
                description="Task 3",
                expected_output="Output 3",
                agent=Agent(role="Agent 3", goal="Goal 3", backstory="Backstory 3"),
            ),
        ]
        return CrewPlanner(tasks, None)

    @pytest.fixture
    def crew_planner_different_llm(self):
        tasks = [
            Task(
                description="Task 1",
                expected_output="Output 1",
                agent=Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1"),
            )
        ]
        planning_agent_llm = "gpt-3.5-turbo"
        return CrewPlanner(tasks, planning_agent_llm)

    def test_handle_crew_planning(self, crew_planner):
        list_of_plans_per_task = [
            PlanPerTask(task="Task1", plan="Plan 1"),
            PlanPerTask(task="Task2", plan="Plan 2"),
            PlanPerTask(task="Task3", plan="Plan 3"),
        ]
        with patch.object(Task, "execute_sync") as execute:
            execute.return_value = TaskOutput(
                description="Description",
                agent="agent",
                pydantic=PlannerTaskPydanticOutput(
                    list_of_plans_per_task=list_of_plans_per_task
                ),
            )
            result = crew_planner._handle_crew_planning()
            assert crew_planner.planning_agent_llm == "gpt-4o-mini"
            assert isinstance(result, PlannerTaskPydanticOutput)
            assert len(result.list_of_plans_per_task) == len(crew_planner.tasks)
            execute.assert_called_once()

    def test_create_planning_agent(self, crew_planner):
        agent = crew_planner._create_planning_agent()
        assert isinstance(agent, Agent)
        assert agent.role == "Task Execution Planner"

    def test_create_planner_task(self, crew_planner):
        planning_agent = Agent(
            role="Planning Agent",
            goal="Plan Step by Step Plan",
            backstory="Master in Planning",
        )
        tasks_summary = "Summary of tasks"
        task = crew_planner._create_planner_task(planning_agent, tasks_summary)

        assert isinstance(task, Task)
        assert task.description.startswith("Based on these tasks summary")
        assert task.agent == planning_agent
        assert (
            task.expected_output
            == "Step by step plan on how the agents can execute their tasks using the available tools with mastery"
        )

    def test_create_tasks_summary(self, crew_planner):
        tasks_summary = crew_planner._create_tasks_summary()
        assert isinstance(tasks_summary, str)
        assert tasks_summary.startswith("\n                Task Number 1 - Task 1")
        assert tasks_summary.endswith('"agent_tools": []\n                ')

    def test_handle_crew_planning_different_llm(self, crew_planner_different_llm):
        with patch.object(Task, "execute_sync") as execute:
            execute.return_value = TaskOutput(
                description="Description",
                agent="agent",
                pydantic=PlannerTaskPydanticOutput(
                    list_of_plans_per_task=[PlanPerTask(task="Task1", plan="Plan 1")]
                ),
            )
            result = crew_planner_different_llm._handle_crew_planning()

            assert crew_planner_different_llm.planning_agent_llm == "gpt-3.5-turbo"
            assert isinstance(result, PlannerTaskPydanticOutput)
            assert len(result.list_of_plans_per_task) == len(
                crew_planner_different_llm.tasks
            )
            execute.assert_called_once()
