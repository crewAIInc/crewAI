"""Types for agent planning and todo tracking."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# Todo status type
TodoStatus = Literal["pending", "running", "completed"]


class PlanStep(BaseModel):
    """A single step in the reasoning plan."""

    step_number: int = Field(description="Step number (1-based)")
    description: str = Field(description="What to do in this step")
    tool_to_use: str | None = Field(
        default=None, description="Tool to use for this step, if any"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Step numbers this step depends on"
    )


class TodoItem(BaseModel):
    """A single todo item representing a step in the execution plan."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    step_number: int = Field(description="Order of this step in the plan (1-based)")
    description: str = Field(description="What needs to be done")
    tool_to_use: str | None = Field(
        default=None, description="Tool to use for this step, if any"
    )
    status: TodoStatus = Field(default="pending", description="Current status")
    depends_on: list[int] = Field(
        default_factory=list, description="Step numbers this depends on"
    )
    result: str | None = Field(
        default=None, description="Result after completion, if any"
    )


class TodoList(BaseModel):
    """Collection of todos for tracking plan execution."""

    items: list[TodoItem] = Field(default_factory=list)

    @property
    def current_todo(self) -> TodoItem | None:
        """Get the currently running todo item."""
        for item in self.items:
            if item.status == "running":
                return item
        return None

    @property
    def next_pending(self) -> TodoItem | None:
        """Get the next pending todo item."""
        for item in self.items:
            if item.status == "pending":
                return item
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all todos are completed."""
        return len(self.items) > 0 and all(
            item.status == "completed" for item in self.items
        )

    @property
    def pending_count(self) -> int:
        """Count of pending todos."""
        return sum(1 for item in self.items if item.status == "pending")

    @property
    def completed_count(self) -> int:
        """Count of completed todos."""
        return sum(1 for item in self.items if item.status == "completed")

    def get_by_step_number(self, step_number: int) -> TodoItem | None:
        """Get a todo by its step number."""
        for item in self.items:
            if item.step_number == step_number:
                return item
        return None

    def mark_running(self, step_number: int) -> None:
        """Mark a todo as running by step number."""
        item = self.get_by_step_number(step_number)
        if item:
            item.status = "running"

    def mark_completed(self, step_number: int, result: str | None = None) -> None:
        """Mark a todo as completed by step number."""
        item = self.get_by_step_number(step_number)
        if item:
            item.status = "completed"
            if result:
                item.result = result

    def _dependencies_satisfied(self, item: TodoItem) -> bool:
        """Check if all dependencies for a todo item are completed.

        Args:
            item: The todo item to check dependencies for.

        Returns:
            True if all dependencies are completed, False otherwise.
        """
        for dep_num in item.depends_on:
            dep = self.get_by_step_number(dep_num)
            if dep is None or dep.status != "completed":
                return False
        return True

    def get_ready_todos(self) -> list[TodoItem]:
        """Get all todos that are ready to execute (pending with satisfied dependencies).

        Returns:
            List of TodoItem objects that can be executed now.
        """
        ready: list[TodoItem] = []
        for item in self.items:
            if item.status != "pending":
                continue
            if self._dependencies_satisfied(item):
                ready.append(item)
        return ready

    @property
    def can_parallelize(self) -> bool:
        """Check if multiple todos can run in parallel.

        Returns:
            True if more than one todo is ready to execute.
        """
        return len(self.get_ready_todos()) > 1

    @property
    def running_count(self) -> int:
        """Count of currently running todos."""
        return sum(1 for item in self.items if item.status == "running")
