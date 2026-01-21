"""CrewAI events system for monitoring and extending agent behavior.

This module provides the event infrastructure that allows users to:
- Monitor agent, task, and crew execution
- Track memory operations and performance
- Build custom logging and analytics
- Extend CrewAI with custom event handlers
- Declare handler dependencies for ordered execution
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.depends import Depends
from crewai.events.event_bus import crewai_event_bus
from crewai.events.handler_graph import CircularDependencyError
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTestResultEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    CrewTrainStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowEvent,
    FlowFinishedEvent,
    FlowPlotEvent,
    FlowStartedEvent,
    HumanFeedbackReceivedEvent,
    HumanFeedbackRequestedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.knowledge_events import (
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeRetrievalStartedEvent,
    KnowledgeSearchQueryFailedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
)
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.events.types.logging_events import (
    AgentLogsExecutionEvent,
    AgentLogsStartedEvent,
)
from crewai.events.types.mcp_events import (
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
)
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalFailedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.events.types.reasoning_events import (
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
    ReasoningEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskEvaluationEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolExecutionErrorEvent,
    ToolSelectionErrorEvent,
    ToolUsageErrorEvent,
    ToolUsageEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    ToolValidateInputErrorEvent,
)


if TYPE_CHECKING:
    from crewai.events.types.agent_events import (
        AgentEvaluationCompletedEvent,
        AgentEvaluationFailedEvent,
        AgentEvaluationStartedEvent,
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        AgentExecutionStartedEvent,
        LiteAgentExecutionCompletedEvent,
        LiteAgentExecutionErrorEvent,
        LiteAgentExecutionStartedEvent,
    )


__all__ = [
    "AgentEvaluationCompletedEvent",
    "AgentEvaluationFailedEvent",
    "AgentEvaluationStartedEvent",
    "AgentExecutionCompletedEvent",
    "AgentExecutionErrorEvent",
    "AgentExecutionStartedEvent",
    "AgentLogsExecutionEvent",
    "AgentLogsStartedEvent",
    "AgentReasoningCompletedEvent",
    "AgentReasoningFailedEvent",
    "AgentReasoningStartedEvent",
    "BaseEventListener",
    "CircularDependencyError",
    "CrewKickoffCompletedEvent",
    "CrewKickoffFailedEvent",
    "CrewKickoffStartedEvent",
    "CrewTestCompletedEvent",
    "CrewTestFailedEvent",
    "CrewTestResultEvent",
    "CrewTestStartedEvent",
    "CrewTrainCompletedEvent",
    "CrewTrainFailedEvent",
    "CrewTrainStartedEvent",
    "Depends",
    "FlowCreatedEvent",
    "FlowEvent",
    "FlowFinishedEvent",
    "FlowPlotEvent",
    "FlowStartedEvent",
    "HumanFeedbackReceivedEvent",
    "HumanFeedbackRequestedEvent",
    "KnowledgeQueryCompletedEvent",
    "KnowledgeQueryFailedEvent",
    "KnowledgeQueryStartedEvent",
    "KnowledgeRetrievalCompletedEvent",
    "KnowledgeRetrievalStartedEvent",
    "KnowledgeSearchQueryFailedEvent",
    "LLMCallCompletedEvent",
    "LLMCallFailedEvent",
    "LLMCallStartedEvent",
    "LLMGuardrailCompletedEvent",
    "LLMGuardrailStartedEvent",
    "LLMStreamChunkEvent",
    "LiteAgentExecutionCompletedEvent",
    "LiteAgentExecutionErrorEvent",
    "LiteAgentExecutionStartedEvent",
    "MCPConnectionCompletedEvent",
    "MCPConnectionFailedEvent",
    "MCPConnectionStartedEvent",
    "MCPToolExecutionCompletedEvent",
    "MCPToolExecutionFailedEvent",
    "MCPToolExecutionStartedEvent",
    "MemoryQueryCompletedEvent",
    "MemoryQueryFailedEvent",
    "MemoryQueryStartedEvent",
    "MemoryRetrievalCompletedEvent",
    "MemoryRetrievalFailedEvent",
    "MemoryRetrievalStartedEvent",
    "MemorySaveCompletedEvent",
    "MemorySaveFailedEvent",
    "MemorySaveStartedEvent",
    "MethodExecutionFailedEvent",
    "MethodExecutionFinishedEvent",
    "MethodExecutionStartedEvent",
    "ReasoningEvent",
    "TaskCompletedEvent",
    "TaskEvaluationEvent",
    "TaskFailedEvent",
    "TaskStartedEvent",
    "ToolExecutionErrorEvent",
    "ToolSelectionErrorEvent",
    "ToolUsageErrorEvent",
    "ToolUsageEvent",
    "ToolUsageFinishedEvent",
    "ToolUsageStartedEvent",
    "ToolValidateInputErrorEvent",
    "crewai_event_bus",
]

_AGENT_EVENT_MAPPING = {
    "AgentEvaluationCompletedEvent": "crewai.events.types.agent_events",
    "AgentEvaluationFailedEvent": "crewai.events.types.agent_events",
    "AgentEvaluationStartedEvent": "crewai.events.types.agent_events",
    "AgentExecutionCompletedEvent": "crewai.events.types.agent_events",
    "AgentExecutionErrorEvent": "crewai.events.types.agent_events",
    "AgentExecutionStartedEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionCompletedEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionErrorEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionStartedEvent": "crewai.events.types.agent_events",
}


def __getattr__(name: str) -> Any:
    """Lazy import for agent events to avoid circular imports."""
    if name in _AGENT_EVENT_MAPPING:
        import importlib

        module_path = _AGENT_EVENT_MAPPING[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
