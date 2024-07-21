from unittest.mock import patch
from crewai.tasks.task_output import TaskOutput

import pytest

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities.planning_handler import CrewPlanner, PlannerTaskPydanticOutput


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
        return CrewPlanner(tasks)

    def test_handle_crew_planning(self, crew_planner):
        with patch.object(Task, "execute_sync") as execute:
            execute.return_value = TaskOutput(
                description="Description",
                agent="agent",
                pydantic=PlannerTaskPydanticOutput(
                    list_of_plans_per_task=["Plan 1", "Plan 2", "Plan 3"]
                ),
            )
            result = crew_planner._handle_crew_planning()

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
