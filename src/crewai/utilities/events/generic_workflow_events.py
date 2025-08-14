from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

@dataclass
class GenericWorkflowEvent:
    """Base class for all generic workflow events."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowStartedEvent(GenericWorkflowEvent):
    """Event indicating the start of a workflow."""
    workflow_id: str
    workflow_name: str
    event_type: str = "workflow_started"

@dataclass
class WorkflowCompletedEvent(GenericWorkflowEvent):
    """Event indicating the completion of a workflow."""
    workflow_id: str
    workflow_name: str
    success: bool
    event_type: str = "workflow_completed"

@dataclass
class TaskStartedEvent(GenericWorkflowEvent):
    """Event indicating the start of a task within a workflow."""
    workflow_id: str
    task_id: str
    task_description: str
    assigned_agent_id: str
    assigned_agent_role: str
    event_type: str = "task_started"

@dataclass
class TaskCompletedEvent(GenericWorkflowEvent):
    """Event indicating the completion of a task within a workflow."""
    workflow_id: str
    task_id: str
    task_description: str
    assigned_agent_id: str
    assigned_agent_role: str
    output: Any
    success: bool
    event_type: str = "task_completed"

@dataclass
class AgentActionOccurredEvent(GenericWorkflowEvent):
    """Event indicating an agent performed an action (e.g., tool usage, LLM call)."""
    workflow_id: str
    agent_id: str
    action_type: str # e.g., "tool_usage", "llm_call", "thought_process"
    action_details: Dict[str, Any]
    event_type: str = "agent_action_occurred"

# You might also consider events for:
# - AgentThoughtEvent
# - ToolUsageEvent
# - InterAgentCommunicationEvent
# ... depending on the granularity you want for the generic events.
