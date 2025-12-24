"""Task configuration model for serializable task configs."""

from datetime import datetime
from typing import Any
import uuid

from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Serializable configuration for a CrewAI Task.

    This model captures all task attributes in a serializable format,
    allowing tasks to be stored, loaded, and managed programmatically.

    Attributes:
        id: Unique identifier for the task config
        name: Optional display name for the task
        description: The task description with optional placeholders
        expected_output: Expected output description (optional for action tasks)
        agent_id: ID of the agent config assigned to this task
        agent_role: Alternative: agent role name for assignment
        action_based: Whether this is an action-based task
        async_execution: Whether to execute asynchronously
        human_input: Whether to request human input after completion
        markdown: Whether output should be markdown formatted
        output_file: Optional file path to save output
        output_json_schema: JSON schema for structured output
        output_pydantic_schema: Pydantic model name for structured output
        guardrail: Guardrail function name or description
        guardrails: List of guardrail configurations
        guardrail_max_retries: Maximum guardrail retries
        context_task_ids: IDs of tasks to use as context
        tool_ids: IDs of tools available for this task
        converter_cls: Custom converter class name
        created_at: When the config was created
        updated_at: When the config was last updated
        tags: Optional tags for categorization
        metadata: Additional custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = Field(default=None, description="Display name for the task")
    description: str = Field(..., description="The task description")
    expected_output: str | None = Field(
        default=None, description="Expected output description"
    )

    # Agent assignment (use one of these)
    agent_id: str | None = Field(
        default=None, description="ID of the assigned agent config"
    )
    agent_role: str | None = Field(
        default=None, description="Role of the agent to assign"
    )

    # Execution behavior
    action_based: bool = Field(
        default=True, description="Whether this is an action-based task"
    )
    async_execution: bool = Field(
        default=False, description="Execute asynchronously"
    )
    human_input: bool = Field(
        default=False, description="Request human input after completion"
    )
    markdown: bool = Field(
        default=False, description="Output should be markdown formatted"
    )

    # Output configuration
    output_file: str | None = Field(
        default=None, description="File path to save output"
    )
    output_json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for structured output"
    )
    output_pydantic_schema: str | None = Field(
        default=None, description="Pydantic model name for structured output"
    )

    # Guardrails
    guardrail: str | None = Field(
        default=None, description="Guardrail function name or description"
    )
    guardrails: list[dict[str, Any]] = Field(
        default_factory=list, description="List of guardrail configurations"
    )
    guardrail_max_retries: int = Field(
        default=3, description="Maximum guardrail retries"
    )

    # Context and tools
    context_task_ids: list[str] = Field(
        default_factory=list, description="IDs of tasks to use as context"
    )
    tool_ids: list[str] = Field(
        default_factory=list, description="IDs of tools available for this task"
    )

    # Advanced options
    converter_cls: str | None = Field(
        default=None, description="Custom converter class name"
    )
    callback: str | None = Field(
        default=None, description="Callback function name"
    )

    # Conditional task settings
    is_conditional: bool = Field(
        default=False, description="Whether this is a conditional task"
    )
    condition: str | None = Field(
        default=None, description="Condition function name for conditional tasks"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata"
    )

    model_config = {"extra": "allow"}

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
