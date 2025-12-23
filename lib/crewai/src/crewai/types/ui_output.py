"""UI-friendly output types for crew execution.

These types provide structured data for rendering agent and task
information in user interfaces, dashboards, and real-time displays.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from crewai.crews.crew_output import CrewOutput


class AgentUIInfo(BaseModel):
    """Agent information for UI display.

    Attributes:
        id: Unique identifier of the agent.
        role: Role of the agent.
        goal: Goal of the agent.
        backstory: Backstory of the agent (optional).
    """

    id: str = ""
    role: str = ""
    goal: str = ""
    backstory: str = ""


class TaskUIInfo(BaseModel):
    """Task information for UI display.

    Attributes:
        id: Unique identifier of the task.
        index: Index of the task (0-based).
        name: Name or short description of the task.
        description: Full description of the task.
        expected_output: Expected output description.
        agent: Agent assigned to the task.
        status: Current status (pending, in_progress, completed, failed).
        start_time: When the task started.
        end_time: When the task completed.
        duration_seconds: Time taken to complete the task.
        output: Task output text.
        output_summary: Truncated summary of the output.
    """

    id: str = ""
    index: int = 0
    name: str = ""
    description: str = ""
    expected_output: str = ""
    agent: AgentUIInfo | None = None
    status: str = "pending"
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    output: str | None = None
    output_summary: str | None = None


class CrewUIInfo(BaseModel):
    """Crew information for UI display.

    Attributes:
        id: Unique identifier of the crew.
        name: Name of the crew.
        process: Process type (sequential, hierarchical, continuous).
        total_tasks: Total number of tasks.
        completed_tasks: Number of completed tasks.
        current_task_index: Index of the currently executing task.
        status: Current status (pending, running, completed, failed).
    """

    id: str = ""
    name: str = ""
    process: str = ""
    total_tasks: int = 0
    completed_tasks: int = 0
    current_task_index: int = 0
    status: str = "pending"


class UIOutput(BaseModel):
    """Complete UI-friendly output from crew execution.

    This class provides a structured view of the crew execution state
    that is suitable for rendering in user interfaces.

    Attributes:
        crew: Crew information.
        agents: List of agents in the crew.
        tasks: List of tasks with their execution state.
        current_agent: Currently executing agent (if any).
        current_task: Currently executing task (if any).
        last_updated: When the output was last updated.
        raw_output: Raw output text from the crew.
        execution_time_seconds: Total execution time.

    Example:
        ```python
        # After crew execution
        ui_output = UIOutput.from_crew_output(crew, crew_output)

        # Display in UI
        for task in ui_output.tasks:
            print(f"{task.name}: {task.status}")
            if task.agent:
                print(f"  Agent: {task.agent.role}")
            if task.output_summary:
                print(f"  Output: {task.output_summary}")
        ```
    """

    crew: CrewUIInfo = Field(default_factory=CrewUIInfo)
    agents: list[AgentUIInfo] = Field(default_factory=list)
    tasks: list[TaskUIInfo] = Field(default_factory=list)
    current_agent: AgentUIInfo | None = None
    current_task: TaskUIInfo | None = None
    last_updated: datetime = Field(default_factory=datetime.now)
    raw_output: str = ""
    execution_time_seconds: float = 0.0

    @classmethod
    def from_crew_output(
        cls,
        crew: Any,
        crew_output: "CrewOutput",
        start_time: datetime | None = None,
    ) -> "UIOutput":
        """Create UIOutput from crew and its output.

        Args:
            crew: The Crew instance.
            crew_output: The CrewOutput from execution.
            start_time: Optional start time for execution timing.

        Returns:
            UIOutput with structured crew execution data.
        """
        end_time = datetime.now()

        # Build agent info
        agents: list[AgentUIInfo] = []
        for agent in crew.agents:
            agents.append(
                AgentUIInfo(
                    id=str(agent.id),
                    role=agent.role,
                    goal=agent.goal or "",
                    backstory=agent.backstory or "",
                )
            )

        # Build task info
        tasks: list[TaskUIInfo] = []
        task_outputs = crew_output.tasks_output if crew_output.tasks_output else []

        for i, task in enumerate(crew.tasks):
            task_output = task_outputs[i] if i < len(task_outputs) else None
            output_str = str(task_output.raw) if task_output else None
            output_summary = (
                output_str[:200] + "..." if output_str and len(output_str) > 200 else output_str
            )

            # Find the agent for this task
            agent_info = None
            if task.agent:
                agent_info = AgentUIInfo(
                    id=str(task.agent.id),
                    role=task.agent.role,
                    goal=task.agent.goal or "",
                    backstory=task.agent.backstory or "",
                )

            tasks.append(
                TaskUIInfo(
                    id=str(task.id),
                    index=i,
                    name=task.name or task.description[:50] if task.description else "",
                    description=task.description or "",
                    expected_output=task.expected_output or "",
                    agent=agent_info,
                    status="completed" if task_output else "pending",
                    output=output_str,
                    output_summary=output_summary,
                )
            )

        # Build crew info
        crew_info = CrewUIInfo(
            id=str(crew.id) if hasattr(crew, "id") else "",
            name=crew.name or "",
            process=str(crew.process.value) if crew.process else "",
            total_tasks=len(crew.tasks),
            completed_tasks=len(task_outputs),
            current_task_index=len(task_outputs) - 1 if task_outputs else 0,
            status="completed",
        )

        execution_time = (
            (end_time - start_time).total_seconds() if start_time else 0.0
        )

        return cls(
            crew=crew_info,
            agents=agents,
            tasks=tasks,
            current_agent=None,
            current_task=None,
            last_updated=end_time,
            raw_output=str(crew_output.raw),
            execution_time_seconds=execution_time,
        )

    def get_task_by_id(self, task_id: str) -> TaskUIInfo | None:
        """Get a task by its ID.

        Args:
            task_id: The task ID to find.

        Returns:
            TaskUIInfo if found, None otherwise.
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_agent_by_id(self, agent_id: str) -> AgentUIInfo | None:
        """Get an agent by its ID.

        Args:
            agent_id: The agent ID to find.

        Returns:
            AgentUIInfo if found, None otherwise.
        """
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_tasks_by_agent(self, agent_id: str) -> list[TaskUIInfo]:
        """Get all tasks assigned to an agent.

        Args:
            agent_id: The agent ID to filter by.

        Returns:
            List of TaskUIInfo for the agent.
        """
        return [task for task in self.tasks if task.agent and task.agent.id == agent_id]

    def get_tasks_by_status(self, status: str) -> list[TaskUIInfo]:
        """Get all tasks with a specific status.

        Args:
            status: The status to filter by (pending, in_progress, completed, failed).

        Returns:
            List of TaskUIInfo with the status.
        """
        return [task for task in self.tasks if task.status == status]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the UIOutput.
        """
        return self.model_dump(mode="json")
