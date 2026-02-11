"""Handles planning and coordination of crew tasks."""

import logging

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task


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
