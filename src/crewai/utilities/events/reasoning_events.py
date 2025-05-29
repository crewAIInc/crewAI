from crewai.utilities.events.base_events import BaseEvent


class AgentReasoningStartedEvent(BaseEvent):
    """Event emitted when an agent starts reasoning about a task."""

    type: str = "agent_reasoning_started"
    agent_role: str
    task_id: str
    attempt: int = 1  # The current reasoning/refinement attempt


class AgentReasoningCompletedEvent(BaseEvent):
    """Event emitted when an agent finishes its reasoning process."""

    type: str = "agent_reasoning_completed"
    agent_role: str
    task_id: str
    plan: str
    ready: bool
    attempt: int = 1
    duration_seconds: float = 0.0  # Time taken for reasoning in seconds


class AgentReasoningFailedEvent(BaseEvent):
    """Event emitted when the reasoning process fails."""

    type: str = "agent_reasoning_failed"
    agent_role: str
    task_id: str
    error: str
    attempt: int = 1


class AgentMidExecutionReasoningStartedEvent(BaseEvent):
    """Event emitted when an agent starts mid-execution reasoning."""

    type: str = "agent_mid_execution_reasoning_started"
    agent_role: str
    task_id: str
    current_step: int
    reasoning_trigger: str  # "interval" or "adaptive"


class AgentMidExecutionReasoningCompletedEvent(BaseEvent):
    """Event emitted when an agent completes mid-execution reasoning."""

    type: str = "agent_mid_execution_reasoning_completed"
    agent_role: str
    task_id: str
    current_step: int
    updated_plan: str
    reasoning_trigger: str
    duration_seconds: float = 0.0  # Time taken for reasoning in seconds


class AgentAdaptiveReasoningDecisionEvent(BaseEvent):
    """Event emitted after the agent decides whether to trigger adaptive reasoning."""

    type: str = "agent_adaptive_reasoning_decision"
    agent_role: str
    task_id: str
    should_reason: bool  # Whether the agent decided to reason
    reasoning: str  # Brief explanation / rationale from the LLM
    reasoning_trigger: str = "adaptive"  # Always adaptive for this event
