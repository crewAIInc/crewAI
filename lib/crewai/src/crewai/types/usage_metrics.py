"""Usage metrics tracking for CrewAI execution.

This module provides models for tracking token usage and request metrics
during crew and agent execution.
"""

from pydantic import BaseModel, Field
from typing_extensions import Self


class UsageMetrics(BaseModel):
    """Track usage metrics for crew execution.

    Attributes:
        total_tokens: Total number of tokens used.
        prompt_tokens: Number of tokens used in prompts.
        cached_prompt_tokens: Number of cached prompt tokens used.
        completion_tokens: Number of tokens used in completions.
        successful_requests: Number of successful requests made.
    """

    total_tokens: int = Field(default=0, description="Total number of tokens used.")
    prompt_tokens: int = Field(
        default=0, description="Number of tokens used in prompts."
    )
    cached_prompt_tokens: int = Field(
        default=0, description="Number of cached prompt tokens used."
    )
    completion_tokens: int = Field(
        default=0, description="Number of tokens used in completions."
    )
    successful_requests: int = Field(
        default=0, description="Number of successful requests made."
    )

    def add_usage_metrics(self, usage_metrics: Self) -> None:
        """Add usage metrics from another UsageMetrics object.

        Args:
            usage_metrics: The usage metrics to add.
        """
        self.total_tokens += usage_metrics.total_tokens
        self.prompt_tokens += usage_metrics.prompt_tokens
        self.cached_prompt_tokens += usage_metrics.cached_prompt_tokens
        self.completion_tokens += usage_metrics.completion_tokens
        self.successful_requests += usage_metrics.successful_requests


class AgentTokenMetrics(BaseModel):
    """Token usage metrics for a specific agent.

    Attributes:
        agent_name: Name/role of the agent
        agent_id: Unique identifier for the agent
        total_tokens: Total tokens used by this agent
        prompt_tokens: Prompt tokens used by this agent
        completion_tokens: Completion tokens used by this agent
        successful_requests: Number of successful LLM requests
    """

    agent_name: str = Field(description="Name/role of the agent")
    agent_id: str | None = Field(default=None, description="Unique identifier for the agent")
    total_tokens: int = Field(default=0, description="Total tokens used by this agent")
    prompt_tokens: int = Field(default=0, description="Prompt tokens used by this agent")
    cached_prompt_tokens: int = Field(default=0, description="Cached prompt tokens used by this agent")
    completion_tokens: int = Field(default=0, description="Completion tokens used by this agent")
    successful_requests: int = Field(default=0, description="Number of successful LLM requests")


class TaskTokenMetrics(BaseModel):
    """Token usage metrics for a specific task.

    Attributes:
        task_name: Name of the task
        task_id: Unique identifier for the task
        agent_name: Name of the agent that executed the task
        total_tokens: Total tokens used for this task
        prompt_tokens: Prompt tokens used for this task
        completion_tokens: Completion tokens used for this task
        successful_requests: Number of successful LLM requests
    """

    task_name: str = Field(description="Name of the task")
    task_id: str | None = Field(default=None, description="Unique identifier for the task")
    agent_name: str = Field(description="Name of the agent that executed the task")
    total_tokens: int = Field(default=0, description="Total tokens used for this task")
    prompt_tokens: int = Field(default=0, description="Prompt tokens used for this task")
    cached_prompt_tokens: int = Field(default=0, description="Cached prompt tokens used for this task")
    completion_tokens: int = Field(default=0, description="Completion tokens used for this task")
    successful_requests: int = Field(default=0, description="Number of successful LLM requests")


class WorkflowTokenMetrics(BaseModel):
    """Complete token usage metrics for a crew workflow.

    Attributes:
        total_tokens: Total tokens used across entire workflow
        prompt_tokens: Total prompt tokens used
        completion_tokens: Total completion tokens used
        successful_requests: Total successful requests
        per_agent: Dictionary mapping agent names to their token metrics
        per_task: Dictionary mapping task names to their token metrics
    """

    total_tokens: int = Field(default=0, description="Total tokens used across entire workflow")
    prompt_tokens: int = Field(default=0, description="Total prompt tokens used")
    cached_prompt_tokens: int = Field(default=0, description="Total cached prompt tokens used")
    completion_tokens: int = Field(default=0, description="Total completion tokens used")
    successful_requests: int = Field(default=0, description="Total successful requests")
    per_agent: dict[str, AgentTokenMetrics] = Field(
        default_factory=dict,
        description="Token metrics per agent"
    )
    per_task: dict[str, TaskTokenMetrics] = Field(
        default_factory=dict,
        description="Token metrics per task"
    )
