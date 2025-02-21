from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Model representing a tool call during execution"""

    name: str
    arguments: Dict[str, Any]
    output: str
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class LLMRequest(BaseModel):
    """Model representing the LLM request details"""

    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Model representing the LLM response details"""

    content: str
    finish_reason: Optional[str] = None


class FlowStepIO(BaseModel):
    """Model representing flow step input/output details"""

    function_name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrewTrace(BaseModel):
    """Model for tracking detailed information about LLM interactions and Flow steps"""

    deployment_instance_id: Optional[str] = Field(
        description="ID of the deployment instance"
    )
    trace_id: str = Field(description="Unique identifier for this trace")
    run_id: str = Field(description="Identifier for the execution run")
    agent_role: Optional[str] = Field(description="Role of the agent")
    task_id: Optional[str] = Field(description="ID of the current task being executed")
    task_name: Optional[str] = Field(description="Name of the current task")
    task_description: Optional[str] = Field(
        description="Description of the current task"
    )
    trace_type: str = Field(description="Type of the trace")
    crew_type: str = Field(description="Type of the crew")
    run_type: str = Field(description="Type of the run")

    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    latency_ms: Optional[int] = None

    # Request/Response for LLM calls
    request: Optional[LLMRequest] = None
    response: Optional[LLMResponse] = None

    # Input/Output for Flow steps
    flow_step: Optional[FlowStepIO] = None

    # Tool usage
    tool_calls: List[ToolCall] = Field(default_factory=list)

    # Metrics
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost: Optional[float] = None

    # Additional metadata
    status: str = "running"  # running, completed, error
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
