"""Event listener that builds a verifiable audit trail for a crew."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.agent_events import (
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
)
from crewai.events.types.knowledge_events import (
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeRetrievalStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
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
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.val.attestors import JSONLVALAttestor
from crewai.val.types import (
    VAL_SPEC_VERSION,
    VALAttestor,
    VALRecord,
    compute_record_hash,
    to_json_dict,
    verify_val_chain,
)


DEFAULT_VAL_EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
    LLMCallStartedEvent,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
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
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryRetrievalStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalFailedEvent,
    MCPConnectionStartedEvent,
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPToolExecutionStartedEvent,
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPConfigFetchFailedEvent,
)


class VALAuditListener(BaseEventListener):
    """Collect and attest crew events as hash-linked VAL records."""

    def __init__(
        self,
        *,
        crew: Any,
        topic_id: str | None = None,
        attestor: VALAttestor | None = None,
        event_types: Sequence[type[BaseEvent]] | None = None,
        spec_version: str = VAL_SPEC_VERSION,
    ) -> None:
        self._crew = crew
        self._topic_id = topic_id
        self._attestor = attestor or JSONLVALAttestor.for_topic(topic_id)
        self._spec_version = spec_version
        self._event_types = tuple(event_types or DEFAULT_VAL_EVENT_TYPES)
        self._records: list[VALRecord] = []
        self._sequence = 0
        self._previous_hash: str | None = None
        self._lock = Lock()
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        for event_type in self._event_types:
            crewai_event_bus.register_handler(event_type, self._handle_event)

    @property
    def records(self) -> list[VALRecord]:
        """Return an immutable snapshot of collected records."""
        with self._lock:
            return list(self._records)

    @property
    def attestor(self) -> VALAttestor:
        """Get the configured attestor backend."""
        return self._attestor

    def verify(self) -> bool:
        """Verify the current local audit chain."""
        return verify_val_chain(self.records)

    def _handle_event(self, source: Any, event: BaseEvent) -> None:
        if not self._belongs_to_wrapped_crew(source, event):
            return

        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "event": event.to_json(),
            "source": self._serialize_source(source),
        }

        with self._lock:
            self._sequence += 1
            record_hash = compute_record_hash(payload, self._previous_hash)
            record = VALRecord(
                spec_version=self._spec_version,
                sequence=self._sequence,
                topic_id=self._topic_id,
                event_type=event.type,
                event_id=event.event_id,
                event_timestamp=event.timestamp.isoformat(),
                previous_hash=self._previous_hash,
                record_hash=record_hash,
                payload=to_json_dict(payload),
                attestor=self._attestor.ledger,
            )
            attestation_id = self._attestor.attest(record)
            finalized_record = record.with_attestation(attestation_id)
            self._records.append(finalized_record)
            self._previous_hash = finalized_record.record_hash

    def _belongs_to_wrapped_crew(self, source: Any, event: BaseEvent) -> bool:
        if source is self._crew:
            return True

        crew_id = str(getattr(self._crew, "id", ""))
        source_id = getattr(source, "id", None)
        if source_id is not None and str(source_id) == crew_id:
            return True

        source_crew = getattr(source, "crew", None)
        if source_crew is self._crew:
            return True

        source_agent = getattr(source, "agent", None)
        if source_agent is not None and getattr(source_agent, "crew", None) is self._crew:
            return True

        task_ids = {str(task.id) for task in getattr(self._crew, "tasks", [])}
        agent_ids = {str(agent.id) for agent in getattr(self._crew, "agents", [])}

        if event.task_id is not None and event.task_id in task_ids:
            return True
        if event.agent_id is not None and event.agent_id in agent_ids:
            return True

        return False

    @staticmethod
    def _serialize_source(source: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class_name": source.__class__.__name__,
            "module": source.__class__.__module__,
        }
        source_id = getattr(source, "id", None)
        if source_id is not None:
            payload["id"] = str(source_id)

        for attr in ("name", "role", "description"):
            value = getattr(source, attr, None)
            if isinstance(value, str) and value:
                payload[attr] = value

        return payload
