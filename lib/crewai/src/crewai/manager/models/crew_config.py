"""Crew configuration model for serializable crew configs."""

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field


class CrewConfig(BaseModel):
    """Serializable configuration for a CrewAI Crew.

    This model captures all crew attributes in a serializable format,
    allowing crews to be stored, loaded, and managed programmatically.

    Attributes:
        id: Unique identifier for the crew config
        name: Display name for the crew
        agent_ids: List of agent config IDs in this crew
        task_ids: List of task config IDs in execution order
        process: Execution process type
        verbose: Whether to enable verbose logging
        cache: Whether to enable caching
        max_rpm: Maximum requests per minute
        stream: Whether to enable streaming output
        memory: Whether to enable memory
        planning: Whether to enable planning
        manager_agent_id: ID of the manager agent config (for hierarchical)
        manager_llm: LLM for the manager agent
        function_calling_llm: LLM for function calling
        planning_llm: LLM for planning
        chat_llm: LLM for chat interactions
        short_term_memory_config: Short-term memory configuration
        long_term_memory_config: Long-term memory configuration
        entity_memory_config: Entity memory configuration
        embedder_config: Embedder configuration
        knowledge_source_ids: Knowledge source config IDs
        output_log_file: Path to log file
        task_callback: Task callback function name
        step_callback: Step callback function name
        created_at: When the config was created
        updated_at: When the config was last updated
        tags: Optional tags for categorization
        metadata: Additional custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="crew", description="Display name for the crew")

    # Agents and Tasks
    agent_ids: list[str] = Field(
        default_factory=list, description="Agent config IDs in this crew"
    )
    task_ids: list[str] = Field(
        default_factory=list, description="Task config IDs in execution order"
    )

    # Process configuration
    process: Literal["sequential", "hierarchical"] = Field(
        default="sequential", description="Execution process type"
    )

    # Behavior flags
    verbose: bool = Field(default=False, description="Enable verbose logging")
    cache: bool = Field(default=True, description="Enable caching")
    max_rpm: int | None = Field(default=None, description="Maximum requests per minute")
    stream: bool = Field(default=False, description="Enable streaming output")
    memory: bool = Field(default=False, description="Enable memory")
    planning: bool = Field(default=False, description="Enable planning")

    # Manager configuration (for hierarchical process)
    manager_agent_id: str | None = Field(
        default=None, description="Manager agent config ID"
    )
    manager_llm: str | None = Field(
        default=None, description="LLM for manager agent"
    )

    # LLM configuration
    function_calling_llm: str | None = Field(
        default=None, description="LLM for function calling"
    )
    planning_llm: str | None = Field(
        default=None, description="LLM for planning"
    )
    chat_llm: str | None = Field(
        default=None, description="LLM for chat interactions"
    )

    # Memory configuration
    short_term_memory_config: dict[str, Any] | None = Field(
        default=None, description="Short-term memory configuration"
    )
    long_term_memory_config: dict[str, Any] | None = Field(
        default=None, description="Long-term memory configuration"
    )
    entity_memory_config: dict[str, Any] | None = Field(
        default=None, description="Entity memory configuration"
    )
    external_memory_config: dict[str, Any] | None = Field(
        default=None, description="External memory configuration"
    )

    # Embedder and knowledge
    embedder_config: dict[str, Any] | None = Field(
        default=None, description="Embedder configuration"
    )
    knowledge_source_ids: list[str] = Field(
        default_factory=list, description="Knowledge source config IDs"
    )

    # Logging and callbacks
    output_log_file: str | None = Field(
        default=None, description="Path to log file"
    )
    task_callback: str | None = Field(
        default=None, description="Task callback function name"
    )
    step_callback: str | None = Field(
        default=None, description="Step callback function name"
    )

    # Prompt customization
    prompt_file: str | None = Field(
        default=None, description="Path to custom prompt file"
    )

    # Streaming configuration
    streaming_config: dict[str, Any] | None = Field(
        default=None, description="Streaming behavior configuration"
    )

    # Tracing
    tracing: bool | None = Field(
        default=None, description="Enable tracing (None=check environment)"
    )

    # Continuous mode settings
    iteration_delay: float = Field(
        default=1.0, description="Delay between continuous iterations"
    )
    max_continuous_runtime: int | None = Field(
        default=None, description="Maximum runtime for continuous mode"
    )
    health_check_interval: int = Field(
        default=60, description="Health check interval in seconds"
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
