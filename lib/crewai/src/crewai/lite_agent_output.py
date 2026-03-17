"""Output class for LiteAgent execution results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.utilities.planning_types import TodoItem
from crewai.utilities.types import LLMMessage


class TodoExecutionResult(BaseModel):
    """Summary of a single todo execution."""

    step_number: int = Field(description="Step number in the plan")
    description: str = Field(description="What the todo was supposed to do")
    tool_used: str | None = Field(
        default=None, description="Tool that was used for this step"
    )
    status: str = Field(description="Final status: completed, failed, pending")
    result: str | None = Field(
        default=None, description="Result or error message from execution"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Step numbers this depended on"
    )


class LiteAgentOutput(BaseModel):
    """Class that represents the result of a LiteAgent execution."""

    model_config = {"arbitrary_types_allowed": True}

    raw: str = Field(description="Raw output of the agent", default="")
    pydantic: BaseModel | None = Field(
        description="Pydantic output of the agent", default=None
    )
    agent_role: str = Field(description="Role of the agent that produced this output")
    usage_metrics: dict[str, Any] | None = Field(
        description="Token usage metrics for this execution", default=None
    )
    messages: list[LLMMessage] = Field(description="Messages of the agent", default=[])

    plan: str | None = Field(
        default=None, description="The execution plan that was generated, if any"
    )
    todos: list[TodoExecutionResult] = Field(
        default_factory=list,
        description="List of todos that were executed with their results",
    )
    replan_count: int = Field(
        default=0, description="Number of times the plan was regenerated"
    )
    last_replan_reason: str | None = Field(
        default=None, description="Reason for the last replan, if any"
    )

    @classmethod
    def from_todo_items(cls, todo_items: list[TodoItem]) -> list[TodoExecutionResult]:
        """Convert TodoItem objects to TodoExecutionResult summaries.

        Args:
            todo_items: List of TodoItem objects from execution.

        Returns:
            List of TodoExecutionResult summaries.
        """
        return [
            TodoExecutionResult(
                step_number=item.step_number,
                description=item.description,
                tool_used=item.tool_to_use,
                status=item.status,
                result=item.result,
                depends_on=item.depends_on,
            )
            for item in todo_items
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert pydantic_output to a dictionary."""
        if self.pydantic:
            return self.pydantic.model_dump()
        return {}

    @property
    def completed_todos(self) -> list[TodoExecutionResult]:
        """Get only the completed todos."""
        return [t for t in self.todos if t.status == "completed"]

    @property
    def failed_todos(self) -> list[TodoExecutionResult]:
        """Get only the failed todos."""
        return [t for t in self.todos if t.status == "failed"]

    @property
    def had_plan(self) -> bool:
        """Check if the agent executed with a plan."""
        return self.plan is not None or len(self.todos) > 0

    def __str__(self) -> str:
        """Return the raw output as a string."""
        return self.raw

    def __repr__(self) -> str:
        """Return a detailed representation including todo summary."""
        parts = [f"LiteAgentOutput(role={self.agent_role!r}"]
        if self.todos:
            completed = len(self.completed_todos)
            total = len(self.todos)
            parts.append(f", todos={completed}/{total} completed")
        if self.replan_count > 0:
            parts.append(f", replans={self.replan_count}")
        parts.append(")")
        return "".join(parts)
