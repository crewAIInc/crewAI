"""Agent state management for long-running tasks."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ToolUsage(BaseModel):
    """Record of a single tool usage."""
    tool_name: str = Field(description="Name of the tool used")
    arguments: Dict[str, Any] = Field(description="Arguments passed to the tool (may be truncated)")
    result_summary: Optional[str] = Field(default=None, description="Brief summary of the tool's result")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the tool was used")
    step_number: int = Field(description="Which execution step this tool was used in")


class AgentState(BaseModel):
    """Persistent state object for agent task execution.

    This state object helps agents maintain coherence during long-running tasks
    by tracking plans, progress, and intermediate results without relying solely
    on conversation history.
    """

    # Core fields
    original_plan: List[str] = Field(
        default_factory=list,
        description="The initial plan from first reasoning pass. Never overwrite unless user requests complete replan"
    )

    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Concrete goals to satisfy for task completion"
    )

    scratchpad: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-defined storage for intermediate results and metadata"
    )

    tool_usage_history: List[ToolUsage] = Field(
        default_factory=list,
        description="Detailed history of tool usage including arguments and results"
    )

    # Additional tracking fields
    task_id: Optional[str] = Field(
        default=None,
        description="ID of the current task being executed"
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this state was created"
    )

    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When this state was last modified"
    )

    steps_completed: int = Field(
        default=0,
        description="Number of execution steps completed"
    )

    def set_original_plan(self, plan: List[str]) -> None:
        """Set the original plan (only if not already set)."""
        if not self.original_plan:
            self.original_plan = plan
            self.last_updated = datetime.now()

    def add_to_scratchpad(self, key: str, value: Any) -> None:
        """Add or update a value in the scratchpad."""
        self.scratchpad[key] = value
        self.last_updated = datetime.now()

    def record_tool_usage(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result_summary: Optional[str] = None,
        max_arg_length: int = 200
    ) -> None:
        """Record a tool usage with truncated arguments.

        Args:
            tool_name: Name of the tool used
            arguments: Arguments passed to the tool
            result_summary: Optional brief summary of the result
            max_arg_length: Maximum length for string arguments before truncation
        """
        # Truncate long string arguments to prevent state bloat
        truncated_args = {}
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > max_arg_length:
                truncated_args[key] = value[:max_arg_length] + "..."
            elif isinstance(value, (list, dict)):
                # For complex types, store a summary
                truncated_args[key] = f"<{type(value).__name__} with {len(value)} items>"
            else:
                truncated_args[key] = value

        tool_usage = ToolUsage(
            tool_name=tool_name,
            arguments=truncated_args,
            result_summary=result_summary,
            step_number=self.steps_completed
        )

        self.tool_usage_history.append(tool_usage)
        self.last_updated = datetime.now()

    def increment_steps(self) -> None:
        """Increment the step counter."""
        self.steps_completed += 1
        self.last_updated = datetime.now()

    def reset(self, task_id: Optional[str] = None) -> None:
        """Reset state for a new task."""
        self.original_plan = []
        self.acceptance_criteria = []
        self.scratchpad = {}
        self.tool_usage_history = []
        self.task_id = task_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.steps_completed = 0

    def to_context_string(self) -> str:
        """Generate a concise string representation for LLM context."""
        context = f"Current State (Step {self.steps_completed}):\n"
        context += f"- Task ID: {self.task_id}\n"

        if self.acceptance_criteria:
            context += "- Acceptance Criteria:\n"
            for criterion in self.acceptance_criteria:
                context += f"  • {criterion}\n"

        if self.original_plan:
            context += "- Plan:\n"
            for i, step in enumerate(self.original_plan, 1):
                context += f"  {i}. {step}\n"

        if self.tool_usage_history:
            context += "- Recent Tool Usage:\n"
            # Show last 5 tool uses
            recent_tools = self.tool_usage_history[-5:]
            for usage in recent_tools:
                context += f"  • Step {usage.step_number}: {usage.tool_name}"
                if usage.arguments:
                    args_preview = ", ".join(f"{k}={v}" for k, v in list(usage.arguments.items())[:2])
                    context += f"({args_preview})"
                context += "\n"

        if self.scratchpad:
            context += "- Scratchpad:\n"
            for key, value in self.scratchpad.items():
                context += f"  • {key}: {value}\n"

        return context

    def get_tools_summary(self) -> Dict[str, Any]:
        """Get a summary of tool usage statistics."""
        if not self.tool_usage_history:
            return {"total_tool_uses": 0, "unique_tools": 0, "tools_by_frequency": {}}

        tool_counts = {}
        for usage in self.tool_usage_history:
            tool_counts[usage.tool_name] = tool_counts.get(usage.tool_name, 0) + 1

        return {
            "total_tool_uses": len(self.tool_usage_history),
            "unique_tools": len(set(usage.tool_name for usage in self.tool_usage_history)),
            "tools_by_frequency": dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True))
        }