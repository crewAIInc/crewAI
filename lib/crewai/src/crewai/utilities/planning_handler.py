"""Handles planning and coordination of crew tasks."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


logger = logging.getLogger(__name__)


class PlanPerTask(BaseModel):
    """Represents a plan for a specific task."""

    task_number: int = Field(
        description="The 1-indexed task number this plan corresponds to",
        ge=1,
    )
    task: str = Field(description="The task for which the plan is created")
    plan: str = Field(
        description="The step by step plan on how the agents can execute their tasks using the available tools with mastery",
    )


class PlannerTaskPydanticOutput(BaseModel):
    """Output format for task planning results."""

    list_of_plans_per_task: list[PlanPerTask] = Field(
        ...,
        description="Step by step plan on how the agents can execute their tasks using the available tools with mastery",
    )


class CrewPlanner:
    """Plans and coordinates the execution of crew tasks.

    Attributes:
        tasks: List of tasks to be planned.
        planning_agent_llm: Optional LLM model for the planning agent.
    """

    def __init__(
        self, tasks: list[Task], planning_agent_llm: str | BaseLLM | None = None
    ) -> None:
        """Initialize CrewPlanner with tasks and optional planning agent LLM.

        Args:
            tasks: List of tasks to be planned.
            planning_agent_llm: Optional LLM model for the planning agent. Defaults to None.
        """
        self.tasks = tasks
        self.planning_agent_llm = planning_agent_llm or "gpt-4o-mini"

    def _handle_crew_planning(self) -> PlannerTaskPydanticOutput:
        """Handles the Crew planning by creating detailed step-by-step plans for each task.

        Returns:
            A PlannerTaskPydanticOutput containing the detailed plans for each task.

        Raises:
            ValueError: If the planning output cannot be obtained.
        """
        planning_agent = self._create_planning_agent()
        tasks_summary = self._create_tasks_summary()

        planner_task = self._create_planner_task(
            planning_agent=planning_agent, tasks_summary=tasks_summary
        )

        result = planner_task.execute_sync()

        if isinstance(result.pydantic, PlannerTaskPydanticOutput):
            return result.pydantic

        raise ValueError("Failed to get the Planning output")

    def _create_planning_agent(self) -> Agent:
        """Creates the planning agent for the crew planning.

        Returns:
            An Agent instance configured for planning tasks.
        """
        return Agent(
            role="Task Execution Planner",
            goal=(
                "Your goal is to create an extremely detailed, step-by-step plan based on the tasks and tools "
                "available to each agent so that they can perform the tasks in an exemplary manner"
            ),
            backstory="Planner agent for crew planning",
            llm=self.planning_agent_llm,
        )

    @staticmethod
    def _create_planner_task(planning_agent: Agent, tasks_summary: str) -> Task:
        """Creates the planner task using the given agent and tasks summary.

        Args:
            planning_agent: The agent responsible for planning.
            tasks_summary: A summary of all tasks to be included in the planning.

        Returns:
            A Task instance configured for planning.
        """
        return Task(
            description=(
                f"Based on these tasks summary: {tasks_summary} \n Create the most descriptive plan based on the tasks "
                "descriptions, tools available, and agents' goals for them to execute their goals with perfection."
            ),
            expected_output="Step by step plan on how the agents can execute their tasks using the available tools with mastery",
            agent=planning_agent,
            output_pydantic=PlannerTaskPydanticOutput,
        )

    @staticmethod
    def _get_agent_knowledge(task: Task) -> list[str]:
        """Safely retrieve knowledge source content from the task's agent.

        Args:
            task: The task containing an agent with potential knowledge sources

        Returns:
            A list of knowledge source strings
        """
        try:
            if task.agent and task.agent.knowledge_sources:
                return [
                    getattr(source, "content", str(source))
                    for source in task.agent.knowledge_sources
                ]
        except AttributeError:
            logger.warning("Error accessing agent knowledge sources")
        return []

    def _handle_crew_replanning(
        self,
        completed_tasks: list[Task],
        completed_outputs: list[TaskOutput],
        remaining_tasks: list[Task],
        deviation_reason: str,
    ) -> PlannerTaskPydanticOutput:
        """Generate revised plans for remaining tasks after a deviation is detected.

        This method is called when a ``ReplanningEvaluator`` determines that a
        completed task's result deviates significantly from the original plan.
        It creates a new plan that accounts for the actual results so far.

        Args:
            completed_tasks: Tasks that have already been executed.
            completed_outputs: Outputs produced by the completed tasks.
            remaining_tasks: Tasks that still need to be executed.
            deviation_reason: Explanation of why replanning was triggered.

        Returns:
            A PlannerTaskPydanticOutput with revised plans for the remaining tasks.

        Raises:
            ValueError: If the replanning output cannot be obtained.
        """
        planning_agent = self._create_planning_agent()
        completed_summary = self._create_completed_tasks_summary(
            completed_tasks, completed_outputs
        )
        remaining_summary = self._create_tasks_summary_for(remaining_tasks)

        replan_task = Task(
            description=(
                "The crew's execution plan needs to be revised because a task result "
                "deviated from the original plan's assumptions.\n\n"
                f"## Reason for Replanning\n{deviation_reason}\n\n"
                f"## Completed Tasks and Their Results\n{completed_summary}\n\n"
                f"## Remaining Tasks That Need New Plans\n{remaining_summary}\n\n"
                "Create revised step-by-step plans for the remaining tasks ONLY, "
                "taking into account what has actually been accomplished so far "
                "and the deviation from the original plan. The plans should adapt "
                "to the real situation rather than following the now-outdated assumptions."
            ),
            expected_output=(
                "Step by step revised plan for each remaining task, "
                "adapted to the actual results so far."
            ),
            agent=planning_agent,
            output_pydantic=PlannerTaskPydanticOutput,
        )

        result = replan_task.execute_sync()

        if isinstance(result.pydantic, PlannerTaskPydanticOutput):
            return result.pydantic

        raise ValueError("Failed to get the Replanning output")

    @staticmethod
    def _create_completed_tasks_summary(
        tasks: list[Task], outputs: list[TaskOutput]
    ) -> str:
        """Create a summary of completed tasks and their actual outputs.

        Args:
            tasks: The completed tasks.
            outputs: The outputs from those tasks.

        Returns:
            A formatted string summarizing completed tasks and results.
        """
        summaries = []
        for idx, (task, output) in enumerate(
            zip(tasks, outputs, strict=False), start=1
        ):
            agent_role = task.agent.role if task.agent else "None"
            summaries.append(
                f"Task {idx} (Agent: {agent_role}):\n"
                f"  Description: {task.description}\n"
                f"  Expected Output: {task.expected_output}\n"
                f"  Actual Result: {output.raw}"
            )
        return "\n\n".join(summaries) if summaries else "No completed tasks."

    @staticmethod
    def _create_tasks_summary_for(tasks: list[Task]) -> str:
        """Create a summary of a subset of tasks (used for remaining tasks).

        Args:
            tasks: The tasks to summarize.

        Returns:
            A formatted string summarizing the tasks.
        """
        summaries = []
        for idx, task in enumerate(tasks, start=1):
            agent_role = task.agent.role if task.agent else "None"
            agent_goal = task.agent.goal if task.agent else "None"
            summaries.append(
                f"Task Number {idx}:\n"
                f'  "task_description": {task.description}\n'
                f'  "task_expected_output": {task.expected_output}\n'
                f'  "agent": {agent_role}\n'
                f'  "agent_goal": {agent_goal}\n'
                f'  "task_tools": {task.tools}\n'
                f'  "agent_tools": {task.agent.tools if task.agent else "None"}'
            )
        return "\n\n".join(summaries) if summaries else "No remaining tasks."

    def _create_tasks_summary(self) -> str:
        """Creates a summary of all tasks.

        Returns:
            A string summarizing all tasks with their details.
        """
        tasks_summary = []
        for idx, task in enumerate(self.tasks):
            knowledge_list = self._get_agent_knowledge(task)
            agent_tools = (
                f"[{', '.join(str(tool) for tool in task.agent.tools)}]"
                if task.agent and task.agent.tools
                else '"agent has no tools"',
                f',\n                "agent_knowledge": "[\\"{knowledge_list[0]}\\"]"'
                if knowledge_list and str(knowledge_list) != "None"
                else "",
            )
            task_summary = f"""
                Task Number {idx + 1} - {task.description}
                "task_description": {task.description}
                "task_expected_output": {task.expected_output}
                "agent": {task.agent.role if task.agent else "None"}
                "agent_goal": {task.agent.goal if task.agent else "None"}
                "task_tools": {task.tools}
                "agent_tools": {"".join(agent_tools)}"""

            tasks_summary.append(task_summary)
        return " ".join(tasks_summary)
