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

    def get_completed_todos(self) -> list[TodoItem]:
        """Get all completed todos.

        Returns:
            List of completed TodoItem objects.
        """
        return [item for item in self.items if item.status == "completed"]

    def get_pending_todos(self) -> list[TodoItem]:
        """Get all pending todos.

        Returns:
            List of pending TodoItem objects.
        """
        return [item for item in self.items if item.status == "pending"]

    def replace_pending_todos(self, new_items: list[TodoItem]) -> None:
        """Replace all pending todos with new items.

        Preserves completed and running todos, replaces only pending ones.
        Used during replanning to swap in a new plan for remaining work.

        Args:
            new_items: The new todo items to replace pending ones.
        """
        non_pending = [item for item in self.items if item.status != "pending"]
        self.items = non_pending + new_items


class StepRefinement(BaseModel):
    """A structured in-place update for a single pending step.

    Returned as part of StepObservation when the Planner learns new
    information that makes a pending step description more specific.
    Applied directly — no second LLM call required.
    """

    step_number: int = Field(description="The step number to update (1-based)")
    new_description: str = Field(
        description="The updated, more specific description for this step"
    )


class StepObservation(BaseModel):
    """Planner's observation after a step execution completes.

    Returned by the PlannerObserver after EVERY step — not just failures.
    The Planner uses this to decide whether to continue, refine, or replan.

    Based on PLAN-AND-ACT (Section 3.3): the Planner observes what the Executor
    did and incorporates new information into the remaining plan.

    Attributes:
        step_completed_successfully: Whether the step achieved its objective.
        key_information_learned: New information revealed by this step
            (e.g., "Found 3 products: A, B, C"). Used to refine upcoming steps.
        remaining_plan_still_valid: Whether pending todos still make sense
            given the new information. True does NOT mean no refinement needed.
        suggested_refinements: Structured in-place updates to pending step
            descriptions. Each entry targets a specific step by number. These
            are applied directly without a second LLM call.
            Example: [{"step_number": 3, "new_description": "Select product B (highest rated)"}]
        needs_full_replan: The remaining plan is fundamentally wrong and must
            be regenerated from scratch. Mutually exclusive with
            remaining_plan_still_valid (if this is True, that should be False).
        replan_reason: Explanation of why a full replan is needed (None if not).
        goal_already_achieved: The overall task goal has been satisfied early.
            No more steps needed — skip remaining todos and finalize.
    """

    step_completed_successfully: bool = Field(
        description="Whether the step achieved what it was asked to do"
    )
    key_information_learned: str = Field(
        default="",
        description="What new information this step revealed",
    )
    remaining_plan_still_valid: bool = Field(
        default=True,
        description="Whether the remaining pending todos still make sense given new information",
    )
    suggested_refinements: list[StepRefinement] | None = Field(
        default=None,
        description=(
            "Structured updates to pending step descriptions based on new information. "
            "Each entry specifies a step_number and new_description. "
            "Applied directly — no separate replan needed."
        ),
    )
    needs_full_replan: bool = Field(
        default=False,
        description="The remaining plan is fundamentally wrong and must be regenerated",
    )
    replan_reason: str | None = Field(
        default=None,
        description="Explanation of why a full replan is needed",
    )
    goal_already_achieved: bool = Field(
        default=False,
        description="The overall task goal has been satisfied early; no more steps needed",
    )
