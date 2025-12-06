"""Tests for the planning handler module."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.tools.base_tool import BaseTool
from crewai.utilities.planning_handler import (
    CrewPlanner,
    PlannerTaskPydanticOutput,
    PlanPerTask,
)


class TestInternalCrewPlanner:
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
            PlanPerTask(task_number=1, task="Task1", plan="Plan 1"),
            PlanPerTask(task_number=2, task="Task2", plan="Plan 2"),
            PlanPerTask(task_number=3, task="Task3", plan="Plan 3"),
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
        assert '"agent_tools": "agent has no tools"' in tasks_summary
        # Knowledge field should not be present when empty
        assert '"agent_knowledge"' not in tasks_summary

    @patch("crewai.knowledge.knowledge.Knowledge.add_sources")
    @patch("crewai.knowledge.storage.knowledge_storage.KnowledgeStorage")
    def test_create_tasks_summary_with_knowledge_and_tools(
        self, mock_storage, mock_add_sources
    ):
        """Test task summary generation with both knowledge and tools present."""

        # Create mock tools with proper string descriptions and structured tool support
        class MockTool(BaseTool):
            name: str
            description: str

            def __init__(self, name: str, description: str):
                tool_data = {"name": name, "description": description}
                super().__init__(**tool_data)

            def __str__(self):
                return self.name

            def __repr__(self):
                return self.name

            def to_structured_tool(self):
                return self

            def _run(self, *args, **kwargs):
                pass

            def _generate_description(self) -> str:
                """Override _generate_description to avoid args_schema handling."""
                return self.description

        tool1 = MockTool("tool1", "Tool 1 description")
        tool2 = MockTool("tool2", "Tool 2 description")

        # Create a task with knowledge and tools
        task = Task(
            description="Task with knowledge and tools",
            expected_output="Expected output",
            agent=Agent(
                role="Test Agent",
                goal="Test Goal",
                backstory="Test Backstory",
                tools=[tool1, tool2],
                knowledge_sources=[
                    StringKnowledgeSource(content="Test knowledge content")
                ],
            ),
        )

        # Create planner with the new task
        planner = CrewPlanner([task], None)
        tasks_summary = planner._create_tasks_summary()

        # Verify task summary content
        assert isinstance(tasks_summary, str)
        assert task.description in tasks_summary
        assert task.expected_output in tasks_summary
        assert '"agent_tools": [tool1, tool2]' in tasks_summary
        assert '"agent_knowledge": "[\\"Test knowledge content\\"]"' in tasks_summary
        assert task.agent.role in tasks_summary
        assert task.agent.goal in tasks_summary

    def test_handle_crew_planning_different_llm(self, crew_planner_different_llm):
        with patch.object(Task, "execute_sync") as execute:
            execute.return_value = TaskOutput(
                description="Description",
                agent="agent",
                pydantic=PlannerTaskPydanticOutput(
                    list_of_plans_per_task=[
                        PlanPerTask(task_number=1, task="Task1", plan="Plan 1")
                    ]
                ),
            )
            result = crew_planner_different_llm._handle_crew_planning()

            assert crew_planner_different_llm.planning_agent_llm == "gpt-3.5-turbo"
            assert isinstance(result, PlannerTaskPydanticOutput)
            assert len(result.list_of_plans_per_task) == len(
                crew_planner_different_llm.tasks
            )
            execute.assert_called_once()

    def test_plan_per_task_requires_task_number(self):
        """Test that PlanPerTask model requires task_number field."""
        with pytest.raises(ValueError):
            PlanPerTask(task="Task1", plan="Plan 1")

    def test_plan_per_task_with_task_number(self):
        """Test PlanPerTask model with task_number field."""
        plan = PlanPerTask(task_number=5, task="Task5", plan="Plan for task 5")
        assert plan.task_number == 5
        assert plan.task == "Task5"
        assert plan.plan == "Plan for task 5"


class TestCrewPlanningIntegration:
    """Tests for Crew._handle_crew_planning integration with task_number matching."""

    def test_crew_planning_with_out_of_order_plans(self):
        """Test that plans are correctly matched to tasks even when returned out of order.

        This test verifies the fix for issue #3953 where plans returned by the LLM
        in a different order than the tasks would be incorrectly assigned.
        """
        agent1 = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")
        agent2 = Agent(role="Agent 2", goal="Goal 2", backstory="Backstory 2")
        agent3 = Agent(role="Agent 3", goal="Goal 3", backstory="Backstory 3")

        task1 = Task(
            description="First task description",
            expected_output="Output 1",
            agent=agent1,
        )
        task2 = Task(
            description="Second task description",
            expected_output="Output 2",
            agent=agent2,
        )
        task3 = Task(
            description="Third task description",
            expected_output="Output 3",
            agent=agent3,
        )

        crew = Crew(
            agents=[agent1, agent2, agent3],
            tasks=[task1, task2, task3],
            planning=True,
        )

        out_of_order_plans = [
            PlanPerTask(task_number=3, task="Task 3", plan=" [PLAN FOR TASK 3]"),
            PlanPerTask(task_number=1, task="Task 1", plan=" [PLAN FOR TASK 1]"),
            PlanPerTask(task_number=2, task="Task 2", plan=" [PLAN FOR TASK 2]"),
        ]

        mock_planner_result = PlannerTaskPydanticOutput(
            list_of_plans_per_task=out_of_order_plans
        )

        with patch.object(
            CrewPlanner, "_handle_crew_planning", return_value=mock_planner_result
        ):
            crew._handle_crew_planning()

        assert "[PLAN FOR TASK 1]" in task1.description
        assert "[PLAN FOR TASK 2]" in task2.description
        assert "[PLAN FOR TASK 3]" in task3.description

        assert "[PLAN FOR TASK 3]" not in task1.description
        assert "[PLAN FOR TASK 1]" not in task2.description
        assert "[PLAN FOR TASK 2]" not in task3.description

    def test_crew_planning_with_missing_plan(self):
        """Test that missing plans are handled gracefully with a warning."""
        agent1 = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")
        agent2 = Agent(role="Agent 2", goal="Goal 2", backstory="Backstory 2")

        task1 = Task(
            description="First task description",
            expected_output="Output 1",
            agent=agent1,
        )
        task2 = Task(
            description="Second task description",
            expected_output="Output 2",
            agent=agent2,
        )

        crew = Crew(
            agents=[agent1, agent2],
            tasks=[task1, task2],
            planning=True,
        )

        original_task1_desc = task1.description
        original_task2_desc = task2.description

        incomplete_plans = [
            PlanPerTask(task_number=1, task="Task 1", plan=" [PLAN FOR TASK 1]"),
        ]

        mock_planner_result = PlannerTaskPydanticOutput(
            list_of_plans_per_task=incomplete_plans
        )

        with patch.object(
            CrewPlanner, "_handle_crew_planning", return_value=mock_planner_result
        ):
            crew._handle_crew_planning()

        assert "[PLAN FOR TASK 1]" in task1.description

        assert task2.description == original_task2_desc

    def test_crew_planning_preserves_original_description(self):
        """Test that planning appends to the original task description."""
        agent = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")

        task = Task(
            description="Original task description",
            expected_output="Output 1",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            planning=True,
        )

        plans = [
            PlanPerTask(task_number=1, task="Task 1", plan=" - Additional plan steps"),
        ]

        mock_planner_result = PlannerTaskPydanticOutput(list_of_plans_per_task=plans)

        with patch.object(
            CrewPlanner, "_handle_crew_planning", return_value=mock_planner_result
        ):
            crew._handle_crew_planning()

        assert "Original task description" in task.description
        assert "Additional plan steps" in task.description

    def test_crew_planning_with_duplicate_task_numbers(self):
        """Test that duplicate task numbers use the first plan and log a warning."""
        agent = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")

        task = Task(
            description="Task description",
            expected_output="Output 1",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            planning=True,
        )

        # Two plans with the same task_number - should use the first one
        duplicate_plans = [
            PlanPerTask(task_number=1, task="Task 1", plan=" [FIRST PLAN]"),
            PlanPerTask(task_number=1, task="Task 1", plan=" [SECOND PLAN]"),
        ]

        mock_planner_result = PlannerTaskPydanticOutput(
            list_of_plans_per_task=duplicate_plans
        )

        with patch.object(
            CrewPlanner, "_handle_crew_planning", return_value=mock_planner_result
        ):
            crew._handle_crew_planning()

        # Should use the first plan, not the second
        assert "[FIRST PLAN]" in task.description
        assert "[SECOND PLAN]" not in task.description
