"""CrewAI events system for monitoring and extending agent behavior.

This module provides the event infrastructure that allows users to:
- Monitor agent, task, and crew execution
- Track memory operations and performance
- Build custom logging and analytics
- Extend CrewAI with custom event handlers
- Declare handler dependencies for ordered execution

Event type classes are lazy-loaded on first access to avoid importing
~12 Pydantic model modules (and their transitive deps) at package init time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.depends import Depends
from crewai.events.event_bus import crewai_event_bus
from crewai.events.handler_graph import CircularDependencyError

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
        MCPConfigFetchFailedEvent,
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
    from crewai.events.types.skill_events import (
        SkillActivatedEvent,
        SkillDiscoveryCompletedEvent,
        SkillDiscoveryStartedEvent,
        SkillEvent,
        SkillLoadFailedEvent,
        SkillLoadedEvent,
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

# Map every event class name → its module path for lazy loading
_LAZY_EVENT_MAPPING: dict[str, str] = {
    # agent_events
    "AgentEvaluationCompletedEvent": "crewai.events.types.agent_events",
    "AgentEvaluationFailedEvent": "crewai.events.types.agent_events",
    "AgentEvaluationStartedEvent": "crewai.events.types.agent_events",
    "AgentExecutionCompletedEvent": "crewai.events.types.agent_events",
    "AgentExecutionErrorEvent": "crewai.events.types.agent_events",
    "AgentExecutionStartedEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionCompletedEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionErrorEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionStartedEvent": "crewai.events.types.agent_events",
    # crew_events
    "CrewKickoffCompletedEvent": "crewai.events.types.crew_events",
    "CrewKickoffFailedEvent": "crewai.events.types.crew_events",
    "CrewKickoffStartedEvent": "crewai.events.types.crew_events",
    "CrewTestCompletedEvent": "crewai.events.types.crew_events",
    "CrewTestFailedEvent": "crewai.events.types.crew_events",
    "CrewTestResultEvent": "crewai.events.types.crew_events",
    "CrewTestStartedEvent": "crewai.events.types.crew_events",
    "CrewTrainCompletedEvent": "crewai.events.types.crew_events",
    "CrewTrainFailedEvent": "crewai.events.types.crew_events",
    "CrewTrainStartedEvent": "crewai.events.types.crew_events",
    # flow_events
    "FlowCreatedEvent": "crewai.events.types.flow_events",
    "FlowEvent": "crewai.events.types.flow_events",
    "FlowFinishedEvent": "crewai.events.types.flow_events",
    "FlowPlotEvent": "crewai.events.types.flow_events",
    "FlowStartedEvent": "crewai.events.types.flow_events",
    "HumanFeedbackReceivedEvent": "crewai.events.types.flow_events",
    "HumanFeedbackRequestedEvent": "crewai.events.types.flow_events",
    "MethodExecutionFailedEvent": "crewai.events.types.flow_events",
    "MethodExecutionFinishedEvent": "crewai.events.types.flow_events",
    "MethodExecutionStartedEvent": "crewai.events.types.flow_events",
    # knowledge_events
    "KnowledgeQueryCompletedEvent": "crewai.events.types.knowledge_events",
    "KnowledgeQueryFailedEvent": "crewai.events.types.knowledge_events",
    "KnowledgeQueryStartedEvent": "crewai.events.types.knowledge_events",
    "KnowledgeRetrievalCompletedEvent": "crewai.events.types.knowledge_events",
    "KnowledgeRetrievalStartedEvent": "crewai.events.types.knowledge_events",
    "KnowledgeSearchQueryFailedEvent": "crewai.events.types.knowledge_events",
    # llm_events
    "LLMCallCompletedEvent": "crewai.events.types.llm_events",
    "LLMCallFailedEvent": "crewai.events.types.llm_events",
    "LLMCallStartedEvent": "crewai.events.types.llm_events",
    "LLMStreamChunkEvent": "crewai.events.types.llm_events",
    # llm_guardrail_events
    "LLMGuardrailCompletedEvent": "crewai.events.types.llm_guardrail_events",
    "LLMGuardrailStartedEvent": "crewai.events.types.llm_guardrail_events",
    # logging_events
    "AgentLogsExecutionEvent": "crewai.events.types.logging_events",
    "AgentLogsStartedEvent": "crewai.events.types.logging_events",
    # mcp_events
    "MCPConfigFetchFailedEvent": "crewai.events.types.mcp_events",
    "MCPConnectionCompletedEvent": "crewai.events.types.mcp_events",
    "MCPConnectionFailedEvent": "crewai.events.types.mcp_events",
    "MCPConnectionStartedEvent": "crewai.events.types.mcp_events",
    "MCPToolExecutionCompletedEvent": "crewai.events.types.mcp_events",
    "MCPToolExecutionFailedEvent": "crewai.events.types.mcp_events",
    "MCPToolExecutionStartedEvent": "crewai.events.types.mcp_events",
    # memory_events
    "MemoryQueryCompletedEvent": "crewai.events.types.memory_events",
    "MemoryQueryFailedEvent": "crewai.events.types.memory_events",
    "MemoryQueryStartedEvent": "crewai.events.types.memory_events",
    "MemoryRetrievalCompletedEvent": "crewai.events.types.memory_events",
    "MemoryRetrievalFailedEvent": "crewai.events.types.memory_events",
    "MemoryRetrievalStartedEvent": "crewai.events.types.memory_events",
    "MemorySaveCompletedEvent": "crewai.events.types.memory_events",
    "MemorySaveFailedEvent": "crewai.events.types.memory_events",
    "MemorySaveStartedEvent": "crewai.events.types.memory_events",
    # reasoning_events
    "AgentReasoningCompletedEvent": "crewai.events.types.reasoning_events",
    "AgentReasoningFailedEvent": "crewai.events.types.reasoning_events",
    "AgentReasoningStartedEvent": "crewai.events.types.reasoning_events",
    "ReasoningEvent": "crewai.events.types.reasoning_events",
    # skill_events
    "SkillActivatedEvent": "crewai.events.types.skill_events",
    "SkillDiscoveryCompletedEvent": "crewai.events.types.skill_events",
    "SkillDiscoveryStartedEvent": "crewai.events.types.skill_events",
    "SkillEvent": "crewai.events.types.skill_events",
    "SkillLoadFailedEvent": "crewai.events.types.skill_events",
    "SkillLoadedEvent": "crewai.events.types.skill_events",
    # task_events
    "TaskCompletedEvent": "crewai.events.types.task_events",
    "TaskEvaluationEvent": "crewai.events.types.task_events",
    "TaskFailedEvent": "crewai.events.types.task_events",
    "TaskStartedEvent": "crewai.events.types.task_events",
    # tool_usage_events
    "ToolExecutionErrorEvent": "crewai.events.types.tool_usage_events",
    "ToolSelectionErrorEvent": "crewai.events.types.tool_usage_events",
    "ToolUsageErrorEvent": "crewai.events.types.tool_usage_events",
    "ToolUsageEvent": "crewai.events.types.tool_usage_events",
    "ToolUsageFinishedEvent": "crewai.events.types.tool_usage_events",
    "ToolUsageStartedEvent": "crewai.events.types.tool_usage_events",
    "ToolValidateInputErrorEvent": "crewai.events.types.tool_usage_events",
}

_extension_exports: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy import for event types and registered extensions."""
    if name in _LAZY_EVENT_MAPPING:
        module_path = _LAZY_EVENT_MAPPING[name]
        module = importlib.import_module(module_path)
        val = getattr(module, name)
        globals()[name] = val  # cache for subsequent access
        return val

    if name in _extension_exports:
        value = _extension_exports[name]
        if isinstance(value, str):
            module_path, _, attr_name = value.rpartition(".")
            if module_path:
                module = importlib.import_module(module_path)
                return getattr(module, attr_name)
            return importlib.import_module(value)
        return value

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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
    "MCPConfigFetchFailedEvent",
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
    "SkillActivatedEvent",
    "SkillDiscoveryCompletedEvent",
    "SkillDiscoveryStartedEvent",
    "SkillEvent",
    "SkillLoadFailedEvent",
    "SkillLoadedEvent",
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
    "_extension_exports",
    "crewai_event_bus",
]
