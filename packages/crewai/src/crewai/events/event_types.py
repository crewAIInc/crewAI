from typing import Union

from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
)
from .types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    CrewTrainStartedEvent,
)
from .types.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from .types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
)
from .types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from .types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from .types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from .types.reasoning_events import (
    AgentReasoningStartedEvent,
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
)
from .types.knowledge_events import (
    KnowledgeRetrievalStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeSearchQueryFailedEvent,
)

from .types.memory_events import (
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryRetrievalStartedEvent,
    MemoryRetrievalCompletedEvent,
)

EventTypes = Union[
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewTestStartedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTrainStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    LiteAgentExecutionCompletedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    FlowStartedEvent,
    FlowFinishedEvent,
    MethodExecutionStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionFailedEvent,
    AgentExecutionErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
    LLMCallStartedEvent,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMStreamChunkEvent,
    LLMGuardrailStartedEvent,
    LLMGuardrailCompletedEvent,
    AgentReasoningStartedEvent,
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    KnowledgeRetrievalStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeSearchQueryFailedEvent,
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryRetrievalStartedEvent,
    MemoryRetrievalCompletedEvent,
]
