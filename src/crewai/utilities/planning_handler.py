from typing import List, Optional

from pydantic import BaseModel

from crewai.agent import Agent
from crewai.task import Task


class PlannerTaskPydanticOutput(BaseModel):
    list_of_plans_per_task: List[str]


class CrewPlanner:
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks

    def _handle_crew_planning(self) -> Optional[BaseModel]:
        """Handles the Crew planning by creating detailed step-by-step plans for each task."""
        planning_agent = self._create_planning_agent()
        tasks_summary = self._create_tasks_summary()

        planner_task = self._create_planner_task(planning_agent, tasks_summary)

        return planner_task.execute_sync().pydantic

    def _create_planning_agent(self) -> Agent:
        """Creates the planning agent for the crew planning."""
        return Agent(
            role="Task Execution Planner",
            goal=(
                "Your goal is to create an extremely detailed, step-by-step plan based on the tasks and tools "
                "available to each agent so that they can perform the tasks in an exemplary manner"
            ),
            backstory="Planner agent for crew planning",
        )

    def _create_planner_task(self, planning_agent: Agent, tasks_summary: str) -> Task:
        """Creates the planner task using the given agent and tasks summary."""
        return Task(
            description=(
                f"Based on these tasks summary: {tasks_summary} \n Create the most descriptive plan based on the tasks "
                "descriptions, tools available, and agents' goals for them to execute their goals with perfection."
            ),
            expected_output="Step by step plan on how the agents can execute their tasks using the available tools with mastery",
            agent=planning_agent,
            output_pydantic=PlannerTaskPydanticOutput,
        )

    def _create_tasks_summary(self) -> str:
        """Creates a summary of all tasks."""
        tasks_summary = []
        for idx, task in enumerate(self.tasks):
            tasks_summary.append(
                f"""
                Task Number {idx + 1} - {task.description}
                "task_description": {task.description}
                "task_expected_output": {task.expected_output}
                "agent": {task.agent.role if task.agent else "None"}
                "agent_goal": {task.agent.goal if task.agent else "None"}
                "task_tools": {task.tools}
                "agent_tools": {task.agent.tools if task.agent else "None"}
                """
            )
        return " ".join(tasks_summary)
