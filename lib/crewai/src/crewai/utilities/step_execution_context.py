"""Context and result types for isolated step execution in Plan-and-Execute architecture.

These types mediate between the AgentExecutor (orchestrator) and StepExecutor (per-step worker).
StepExecutionContext carries only final results from dependencies — never LLM message histories.
StepResult carries only the outcome of a step — never internal execution traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepExecutionContext:
    """Immutable context passed to a StepExecutor for a single todo.

    Contains only the information the Executor needs to complete one step:
    the task description, goal, and final results from dependency steps.
    No LLM message history, no execution traces, no shared mutable state.

    Attributes:
        task_description: The original task description (from Task or kickoff input).
        task_goal: The expected output / goal of the overall task.
        dependency_results: Mapping of step_number → final result string
            for all completed dependencies of the current step.
    """

    task_description: str
    task_goal: str
    dependency_results: dict[int, str] = field(default_factory=dict)

    def get_dependency_result(self, step_number: int) -> str | None:
        """Get the final result of a dependency step.

        Args:
            step_number: The step number to look up.

        Returns:
            The result string if available, None otherwise.
        """
        return self.dependency_results.get(step_number)


@dataclass
class StepResult:
    """Result returned by a StepExecutor after executing a single todo.

    Contains the final outcome and metadata for debugging/metrics.
    Tool call details are for audit logging only — they are NOT passed
    to subsequent steps or the Planner.

    Attributes:
        success: Whether the step completed successfully.
        result: The final output string from the step.
        error: Error message if the step failed (None on success).
        tool_calls_made: List of tool names invoked (for debugging/logging only).
        execution_time: Wall-clock time in seconds for the step execution.
    """

    success: bool
    result: str
    error: str | None = None
    tool_calls_made: list[str] = field(default_factory=list)
    execution_time: float = 0.0
