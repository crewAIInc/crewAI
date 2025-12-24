"""Response models for CrewManager operations."""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class OperationResult(BaseModel, Generic[T]):
    """Result of a single CRUD operation.

    Attributes:
        success: Whether the operation succeeded
        data: The resulting data (if successful)
        error: Error message (if failed)
        error_code: Error code for programmatic handling
        timestamp: When the operation completed
    """

    success: bool = Field(..., description="Whether the operation succeeded")
    data: T | None = Field(default=None, description="Resulting data")
    error: str | None = Field(default=None, description="Error message")
    error_code: str | None = Field(default=None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def ok(cls, data: T) -> "OperationResult[T]":
        """Create a successful result."""
        return cls(success=True, data=data)

    @classmethod
    def fail(
        cls, error: str, error_code: str | None = None
    ) -> "OperationResult[T]":
        """Create a failed result."""
        return cls(success=False, error=error, error_code=error_code)


class ListResult(BaseModel, Generic[T]):
    """Result of a list operation with pagination.

    Attributes:
        items: List of items for the current page
        total: Total number of items across all pages
        offset: Current offset
        limit: Items per page
        has_more: Whether there are more items
    """

    items: list[T] = Field(default_factory=list, description="Items in current page")
    total: int = Field(..., description="Total items count")
    offset: int = Field(default=0, description="Current offset")
    limit: int = Field(default=50, description="Items per page")
    has_more: bool = Field(default=False, description="More items available")

    @classmethod
    def from_list(
        cls,
        items: list[T],
        total: int,
        offset: int = 0,
        limit: int = 50,
    ) -> "ListResult[T]":
        """Create a list result from items."""
        return cls(
            items=items,
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + len(items)) < total,
        )


class ExecutionProgress(BaseModel):
    """Progress information during crew execution.

    Attributes:
        crew_id: ID of the executing crew
        status: Current execution status
        current_task_index: Index of the current task
        total_tasks: Total number of tasks
        current_task_name: Name of the current task
        current_agent_role: Role of the current agent
        elapsed_seconds: Execution time so far
    """

    crew_id: str = Field(..., description="Crew ID")
    status: str = Field(..., description="Execution status")
    current_task_index: int = Field(default=0, description="Current task index")
    total_tasks: int = Field(default=0, description="Total tasks")
    current_task_name: str | None = Field(default=None)
    current_agent_role: str | None = Field(default=None)
    elapsed_seconds: float = Field(default=0.0)


class TaskOutputSummary(BaseModel):
    """Summary of a task's output.

    Attributes:
        task_id: ID of the task
        task_name: Name of the task
        agent_role: Role of the agent that executed
        raw: Raw output text
        success: Whether execution succeeded
    """

    task_id: str = Field(..., description="Task ID")
    task_name: str | None = Field(default=None)
    agent_role: str | None = Field(default=None)
    raw: str = Field(default="")
    success: bool = Field(default=True)


class ExecutionResult(BaseModel):
    """Result of a crew execution.

    Attributes:
        success: Whether execution succeeded
        crew_id: ID of the executed crew
        raw_output: Final raw output
        pydantic_output: Structured pydantic output (if any)
        json_output: Structured JSON output (if any)
        task_outputs: Outputs from each task
        token_usage: Token usage statistics
        execution_time_seconds: Total execution time
        error: Error message (if failed)
    """

    success: bool = Field(..., description="Whether execution succeeded")
    crew_id: str = Field(..., description="Crew ID")
    raw_output: str = Field(default="", description="Final raw output")
    pydantic_output: Any | None = Field(default=None, description="Pydantic output")
    json_output: dict[str, Any] | None = Field(default=None, description="JSON output")
    task_outputs: list[TaskOutputSummary] = Field(
        default_factory=list, description="Task outputs"
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict, description="Token usage"
    )
    execution_time_seconds: float = Field(default=0.0, description="Execution time")
    error: str | None = Field(default=None, description="Error message")

    @classmethod
    def from_crew_output(
        cls,
        crew_id: str,
        crew_output: Any,
        execution_time: float,
    ) -> "ExecutionResult":
        """Create from a CrewOutput instance."""
        task_summaries = []
        for task_output in getattr(crew_output, "tasks_output", []):
            task_summaries.append(
                TaskOutputSummary(
                    task_id=str(getattr(task_output, "task_id", "")),
                    task_name=getattr(task_output, "name", None),
                    agent_role=getattr(task_output, "agent", None),
                    raw=getattr(task_output, "raw", ""),
                    success=getattr(task_output, "execution_success", True),
                )
            )

        token_usage = {}
        if hasattr(crew_output, "token_usage") and crew_output.token_usage:
            token_usage = crew_output.token_usage.model_dump()

        return cls(
            success=True,
            crew_id=crew_id,
            raw_output=getattr(crew_output, "raw", ""),
            pydantic_output=getattr(crew_output, "pydantic", None),
            json_output=getattr(crew_output, "json_dict", None),
            task_outputs=task_summaries,
            token_usage=token_usage,
            execution_time_seconds=execution_time,
        )

    @classmethod
    def from_error(cls, crew_id: str, error: str) -> "ExecutionResult":
        """Create a failed result from an error."""
        return cls(
            success=False,
            crew_id=crew_id,
            error=error,
        )
