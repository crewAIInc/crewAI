"""Trace collector that subscribes to the event bus.

One ``TraceCollector`` per Agent instance. It keeps a per-(agent, task)
in-flight trace, identifies which events belong to its agent, and
finalizes the trace on completion — auto-grading and persisting it.
"""

from __future__ import annotations

import atexit
from datetime import UTC, datetime
import logging
import threading
from typing import TYPE_CHECKING, Any

from crewai.events.event_bus import CrewAIEventsBus, crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.skills.self_improve.auto_grade import grade_trace
from crewai.skills.self_improve.models import RunTrace, ToolCallRecord
from crewai.skills.self_improve.storage import TraceStore


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent


_OUTPUT_TRUNCATE = 4000
_ARGS_TRUNCATE = 200
_FLUSH_TIMEOUT = 10.0

logger = logging.getLogger(__name__)

# ``Agent.kickoff()`` can return before the bus's thread-pool handlers drain;
# without this hook a script that exits immediately would lose the just-
# finalized trace. ``flush()`` is a no-op when no events are pending, so it's
# safe to register at module import time even if no agent ever opts in.
atexit.register(lambda: crewai_event_bus.flush(timeout=_FLUSH_TIMEOUT))


def _truncate(value: Any, limit: int) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else repr(value)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


class TraceCollector:
    """Captures one ``RunTrace`` per agent execution by subscribing to events.

    Lifecycle:
        - ``attach(bus)`` registers handlers on the global event bus.
        - On ``AgentExecutionStartedEvent`` for our agent, a new trace begins.
        - ``ToolUsage*`` events for our agent append ``ToolCallRecord``s.
        - On ``AgentExecutionCompletedEvent`` / ``AgentExecutionErrorEvent``
          for our agent, the trace is auto-graded and persisted.

    The collector is intentionally tolerant: a missing Started event
    (e.g. because the agent was already executing when ``attach`` ran) just
    skips that trace. Tool events without a current trace are ignored.
    """

    def __init__(
        self,
        agent: BaseAgent,
        store: TraceStore | None = None,
    ) -> None:
        self._agent = agent
        self._store = store or TraceStore()
        self._current: RunTrace | None = None
        self._tool_started_at: dict[str, datetime] = {}
        self._attached = False
        # Bus dispatches handlers on a thread pool, and Agent.kickoff() +
        # LiteAgent each emit Started/Completed for the same logical run, so
        # the lock + ``self._current is not None`` check serializes the
        # "create or skip" decision and dedupes the second half of the pair.
        self._lock = threading.RLock()

    @property
    def current_trace(self) -> RunTrace | None:
        """The in-flight trace, if an agent execution is active."""
        return self._current

    def _is_my_agent(self, event_agent: Any) -> bool:
        if event_agent is None:
            return False
        if event_agent is self._agent:
            return True
        agent_id = getattr(event_agent, "id", None)
        my_id = getattr(self._agent, "id", None)
        return bool(agent_id is not None and my_id is not None and agent_id == my_id)

    def _is_my_id(self, event_agent_id: str | None) -> bool:
        if not event_agent_id:
            return False
        my_id = getattr(self._agent, "id", None)
        return bool(my_id is not None and str(my_id) == str(event_agent_id))

    def attach(self, bus: CrewAIEventsBus) -> None:
        """Register event handlers. Idempotent."""
        if self._attached:
            return
        self._attached = True

        @bus.on(AgentExecutionStartedEvent)
        def _on_started(_source: Any, event: AgentExecutionStartedEvent) -> None:
            if not self._is_my_agent(event.agent):
                return
            with self._lock:
                if self._current is not None:
                    return  # duplicate Started for an in-flight trace
                task = event.task
                self._current = RunTrace(
                    agent_id=str(getattr(self._agent, "id", "") or ""),
                    agent_role=self._agent.role,
                    agent_goal=getattr(self._agent, "goal", "") or "",
                    task_id=str(getattr(task, "id", "") or "") or None,
                    task_description=getattr(task, "description", None),
                    loaded_skills=self._collect_loaded_skills(),
                )

        @bus.on(ToolUsageStartedEvent)
        def _on_tool_started(_source: Any, event: ToolUsageStartedEvent) -> None:
            if not self._is_my_id(event.agent_id):
                return
            with self._lock:
                if self._current is None:
                    return
                self._tool_started_at[event.tool_name] = datetime.now(UTC)

        @bus.on(ToolUsageFinishedEvent)
        def _on_tool_finished(_source: Any, event: ToolUsageFinishedEvent) -> None:
            if not self._is_my_id(event.agent_id):
                return
            with self._lock:
                if self._current is None:
                    return
                self._current.tool_calls.append(
                    ToolCallRecord(
                        name=event.tool_name,
                        args_summary=_truncate(event.tool_args, _ARGS_TRUNCATE),
                        ok=True,
                        duration_ms=self._duration_ms(
                            event.tool_name, event.finished_at
                        ),
                    )
                )
                self._tool_started_at.pop(event.tool_name, None)

        @bus.on(ToolUsageErrorEvent)
        def _on_tool_error(_source: Any, event: ToolUsageErrorEvent) -> None:
            if not self._is_my_id(event.agent_id):
                return
            with self._lock:
                if self._current is None:
                    return
                self._current.tool_calls.append(
                    ToolCallRecord(
                        name=event.tool_name,
                        args_summary=_truncate(event.tool_args, _ARGS_TRUNCATE),
                        ok=False,
                        error=_truncate(event.error, _ARGS_TRUNCATE),
                        duration_ms=self._duration_ms(
                            event.tool_name, datetime.now(UTC)
                        ),
                    )
                )
                self._tool_started_at.pop(event.tool_name, None)

        @bus.on(AgentExecutionCompletedEvent)
        def _on_completed(_source: Any, event: AgentExecutionCompletedEvent) -> None:
            if not self._is_my_agent(event.agent):
                return
            with self._lock:
                if self._current is None:
                    return
                self._current.output_summary = _truncate(
                    event.output, _OUTPUT_TRUNCATE
                )
                self._finalize_locked()

        @bus.on(AgentExecutionErrorEvent)
        def _on_error(_source: Any, event: AgentExecutionErrorEvent) -> None:
            if not self._is_my_agent(event.agent):
                return
            with self._lock:
                if self._current is None:
                    return
                self._current.error = _truncate(event.error, _OUTPUT_TRUNCATE)
                self._finalize_locked()

        @bus.on(LiteAgentExecutionStartedEvent)
        def _on_lite_started(
            _source: Any, event: LiteAgentExecutionStartedEvent
        ) -> None:
            if not self._is_my_id(event.agent_info.get("id")):
                return
            with self._lock:
                if self._current is not None:
                    return  # duplicate Started for an in-flight trace
                messages = event.messages
                if isinstance(messages, list):
                    task_desc = " ".join(
                        str(m.get("content", ""))
                        for m in messages
                        if isinstance(m, dict)
                    )
                else:
                    task_desc = str(messages)
                self._current = RunTrace(
                    agent_id=str(getattr(self._agent, "id", "") or ""),
                    agent_role=self._agent.role,
                    agent_goal=getattr(self._agent, "goal", "") or "",
                    task_description=_truncate(task_desc, _OUTPUT_TRUNCATE),
                    loaded_skills=self._collect_loaded_skills(),
                )

        @bus.on(LiteAgentExecutionCompletedEvent)
        def _on_lite_completed(
            _source: Any, event: LiteAgentExecutionCompletedEvent
        ) -> None:
            if not self._is_my_id(event.agent_info.get("id")):
                return
            with self._lock:
                if self._current is None:
                    return
                self._current.output_summary = _truncate(
                    event.output, _OUTPUT_TRUNCATE
                )
                self._finalize_locked()

        @bus.on(LiteAgentExecutionErrorEvent)
        def _on_lite_error(
            _source: Any, event: LiteAgentExecutionErrorEvent
        ) -> None:
            if not self._is_my_id(event.agent_info.get("id")):
                return
            with self._lock:
                if self._current is None:
                    return
                self._current.error = _truncate(event.error, _OUTPUT_TRUNCATE)
                self._finalize_locked()

    def _duration_ms(self, tool_name: str, finished_at: datetime) -> int | None:
        started = self._tool_started_at.get(tool_name)
        if started is None:
            return None
        return max(0, int((finished_at - started).total_seconds() * 1000))

    def _collect_loaded_skills(self) -> list[str]:
        skills = getattr(self._agent, "skills", None) or []
        names: list[str] = []
        for s in skills:
            name = getattr(s, "name", None)
            if isinstance(name, str):
                names.append(name)
        return names

    def _finalize_locked(self) -> None:
        """Caller holds ``self._lock``."""
        trace = self._current
        if trace is None:
            return
        self._current = None
        self._tool_started_at.clear()
        trace.ended_at = datetime.now(UTC)
        trace.outcome = grade_trace(trace)
        try:
            self._store.save(trace)
        except OSError:
            logger.exception(
                "Failed to persist run trace for role %s", trace.agent_role
            )
