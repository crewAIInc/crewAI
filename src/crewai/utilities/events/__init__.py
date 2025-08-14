# Existing imports (keep these as they are used by other parts of CrewAI)
from .agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    AgentThoughtEvent,
)
from .base_events import BaseEvent
from .crew_events import CrewKickoffCompletedEvent, CrewKickoffStartedEvent
from .llm_events import LLMCallCompletedEvent, LLMCallStartedEvent
from .memory_events import MemoryRetrievalCompletedEvent, MemoryRetrievalStartedEvent
from .task_events import TaskCompletedEvent, TaskStartedEvent
from .tool_events import (
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolSelectionErrorEvent,
    ToolUsageEvent,
    ToolValidateInputErrorEvent,
)

# Our crypto events (keep these)
from .crypto_events import (
    CryptographicCommitmentCreatedEvent,
    CryptographicValidationCompletedEvent,
    CryptographicWorkflowAuditEvent,
    CryptographicEscrowTransactionEvent,
)

# Our crypto listener (keep this)
from .listeners.crypto_listener import CrewAICryptographicTraceListener

# NEW: Import the CrewAI Event Adapter
from .crewai_event_adapter import CrewAIEventAdapter

# NEW: Import generic workflow events
from .generic_workflow_events import (
    GenericWorkflowEvent,
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    TaskStartedEvent as GenericTaskStartedEvent, # Alias to avoid name collision with CrewAI's TaskStartedEvent
    TaskCompletedEvent as GenericTaskCompletedEvent, # Alias to avoid name collision with CrewAI's TaskCompletedEvent
    AgentActionOccurredEvent,
)

# events (keep this)
from .event_listener import EventListener
from .crewai_event_bus import crewai_event_bus

# __all__ (adjust this)
__all__ = [
    "BaseEvent",
    "AgentExecutionCompletedEvent",
    "AgentExecutionStartedEvent",
    "AgentThoughtEvent",
    "CrewKickoffCompletedEvent",
    "CrewKickoffStartedEvent",
    "LLMCallCompletedEvent",
    "LLMCallStartedEvent",
    "MemoryRetrievalCompletedEvent",
    "MemoryRetrievalStartedEvent",
    "TaskCompletedEvent",
    "TaskStartedEvent",
    "ToolExecutionCompletedEvent",
    "ToolExecutionStartedEvent",
    "ToolSelectionErrorEvent",
    "ToolUsageEvent",
    "ToolValidateInputErrorEvent",
    "CryptographicCommitmentCreatedEvent",
    "CryptographicValidationCompletedEvent",
    "CryptographicWorkflowAuditEvent",
    "CryptographicEscrowTransactionEvent",
    "CrewAICryptographicTraceListener",
    "EventListener",
    "crewai_event_bus",
    # NEW: Add the adapter and generic events
    "CrewAIEventAdapter",
    "GenericWorkflowEvent",
    "WorkflowStartedEvent",
    "WorkflowCompletedEvent",
    "GenericTaskStartedEvent", # Exported with alias
    "GenericTaskCompletedEvent", # Exported with alias
    "AgentActionOccurredEvent",
]