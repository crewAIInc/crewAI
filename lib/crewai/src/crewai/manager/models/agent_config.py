"""Agent configuration model for serializable agent configs."""

from datetime import datetime
from typing import Any
import uuid

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Serializable configuration for a CrewAI Agent.

    This model captures all agent attributes in a serializable format,
    allowing agents to be stored, loaded, and managed programmatically.

    Attributes:
        id: Unique identifier for the agent config
        name: Optional display name for the agent
        role: The role of the agent in the crew
        goal: The objective the agent is trying to achieve
        backstory: Background context for the agent's persona
        llm: LLM model identifier (e.g., "gpt-4", "anthropic/claude-3")
        function_calling_llm: Optional separate LLM for function calling
        tool_ids: List of tool config IDs this agent can use
        max_iter: Maximum iterations for task execution
        max_rpm: Maximum requests per minute
        max_tokens: Maximum tokens for responses
        max_execution_time: Maximum execution time in seconds
        max_retry_limit: Maximum retries on error
        verbose: Whether to enable verbose logging
        cache: Whether to enable caching
        allow_delegation: Whether the agent can delegate to others
        allow_code_execution: Whether the agent can execute code
        code_execution_mode: Mode for code execution ("safe" or "unsafe")
        multimodal: Whether the agent supports multimodal input
        reasoning: Whether the agent should use reasoning before execution
        max_reasoning_attempts: Maximum reasoning attempts
        inject_date: Whether to inject current date into tasks
        date_format: Format string for date injection
        respect_context_window: Whether to keep messages under context window
        use_system_prompt: Whether to use system prompts
        system_template: Custom system prompt template
        prompt_template: Custom prompt template
        response_template: Custom response template
        knowledge_sources: List of knowledge source configurations
        embedder_config: Configuration for the embedder
        created_at: When the config was created
        updated_at: When the config was last updated
        tags: Optional tags for categorization
        metadata: Additional custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = Field(default=None, description="Display name for the agent")
    role: str = Field(..., description="The role of the agent")
    goal: str = Field(..., description="The objective of the agent")
    backstory: str = Field(..., description="Background context for the agent")

    # LLM Configuration
    llm: str | None = Field(
        default=None, description="LLM model identifier (e.g., 'gpt-4')"
    )
    function_calling_llm: str | None = Field(
        default=None, description="Separate LLM for function calling"
    )

    # Tools
    tool_ids: list[str] = Field(
        default_factory=list, description="List of tool config IDs"
    )

    # Execution limits
    max_iter: int = Field(default=25, description="Maximum iterations")
    max_rpm: int | None = Field(default=None, description="Maximum requests per minute")
    max_tokens: int | None = Field(default=None, description="Maximum tokens")
    max_execution_time: int | None = Field(
        default=None, description="Maximum execution time in seconds"
    )
    max_retry_limit: int = Field(default=2, description="Maximum retries on error")

    # Behavior flags
    verbose: bool = Field(default=False, description="Enable verbose logging")
    cache: bool = Field(default=True, description="Enable caching")
    allow_delegation: bool = Field(default=False, description="Allow delegation")
    allow_code_execution: bool = Field(default=False, description="Allow code execution")
    code_execution_mode: str = Field(
        default="safe", description="Code execution mode: 'safe' or 'unsafe'"
    )

    # Advanced features
    multimodal: bool = Field(default=False, description="Support multimodal input")
    reasoning: bool = Field(default=False, description="Use reasoning before execution")
    max_reasoning_attempts: int | None = Field(
        default=None, description="Maximum reasoning attempts"
    )
    inject_date: bool = Field(default=False, description="Inject current date")
    date_format: str = Field(default="%Y-%m-%d", description="Date format string")
    respect_context_window: bool = Field(
        default=True, description="Keep messages under context window"
    )
    use_system_prompt: bool = Field(default=True, description="Use system prompts")

    # Templates
    system_template: str | None = Field(default=None, description="System prompt template")
    prompt_template: str | None = Field(default=None, description="Prompt template")
    response_template: str | None = Field(default=None, description="Response template")

    # Knowledge
    knowledge_source_ids: list[str] = Field(
        default_factory=list, description="Knowledge source config IDs"
    )
    embedder_config: dict[str, Any] | None = Field(
        default=None, description="Embedder configuration"
    )

    # MCP servers
    mcp_server_ids: list[str] = Field(
        default_factory=list, description="MCP server config IDs"
    )

    # Guardrails
    guardrail: str | None = Field(
        default=None, description="Guardrail function name or description"
    )
    guardrail_max_retries: int = Field(
        default=3, description="Maximum guardrail retries"
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
