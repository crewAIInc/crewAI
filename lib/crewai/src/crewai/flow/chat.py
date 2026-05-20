"""Transport-agnostic chat session bridge for conversational flows."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.flow.conversation import (
    get_conversation_messages,
    get_conversational_config,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.flow.flow import Flow
    from crewai.llms.base_llm import BaseLLM
    from crewai.types.streaming import FlowStreamingOutput


ChatMessageType = Literal[
    "user_message",
    "assistant_delta",
    "assistant_done",
    "turn_started",
    "turn_finished",
    "error",
    "tool_started",
    "tool_finished",
]


class ChatMessage(BaseModel):
    """Versioned wire format for chat UIs (WebSocket, SSE, webhooks)."""

    version: str = "1"
    type: ChatMessageType
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    seq: int | None = None


@dataclass
class TurnResult:
    """Outcome of a single conversational turn."""

    session_id: str
    output: Any
    intent: str | None = None
    messages: list[LLMMessage] = field(default_factory=list)
    streaming: FlowStreamingOutput | None = None


class ChatSession:
    """Wraps ``Flow.kickoff`` for one chat session (``state.id``)."""

    def __init__(
        self,
        flow: Flow[Any],
        session_id: str | None = None,
        *,
        intents: Sequence[str] | None = None,
        intent_llm: str | BaseLLM | None = None,
        on_event: Callable[[ChatMessage], None] | None = None,
    ) -> None:
        self._flow = flow
        self._session_id = session_id or str(uuid4())
        self._intents = list(intents) if intents else None
        self._intent_llm = intent_llm
        self._on_event = on_event
        self._seq = 0
        self._bridge: ConversationEventBridge | None = None
        config = get_conversational_config(flow)
        if config is not None and config.defer_trace_finalization:
            flow.defer_trace_finalization = True
        if on_event is not None:
            self._bridge = ConversationEventBridge(
                session_id=self._session_id,
                handler=on_event,
            )
            self._bridge.register()

    @property
    def session_id(self) -> str:
        return self._session_id

    def handle_turn(
        self,
        user_message: str,
        *,
        stream: bool | None = None,
    ) -> TurnResult:
        """Run one conversational turn and return output plus message history."""
        self._emit("turn_started", {"user_message": user_message})
        use_stream = stream if stream is not None else bool(self._flow.stream)

        try:
            result = self._flow.kickoff(
                user_message=user_message,
                session_id=self._session_id,
                intents=self._intents,
                intent_llm=self._intent_llm,
            )
        except Exception as exc:
            self._emit("error", {"message": str(exc)})
            raise

        streaming = None
        output: Any = result
        if use_stream and hasattr(result, "__iter__"):
            from crewai.types.streaming import FlowStreamingOutput

            if isinstance(result, FlowStreamingOutput):
                streaming = result
                for chunk in result:
                    text = getattr(chunk, "content", None) or str(chunk)
                    self._emit("assistant_delta", {"chunk": text})
                output = result.result
            else:
                for chunk in result:
                    text = getattr(chunk, "content", None) or str(chunk)
                    self._emit("assistant_delta", {"chunk": text})

        intent = None
        state = self._flow.state
        if hasattr(state, "last_intent"):
            intent = getattr(state, "last_intent", None)
        elif isinstance(state, dict):
            intent = state.get("last_intent")

        messages = get_conversation_messages(self._flow)
        self._emit(
            "assistant_done",
            {"output": output, "intent": intent},
        )
        self._emit("turn_finished", {"output": output})

        return TurnResult(
            session_id=self._session_id,
            output=output,
            intent=intent,
            messages=messages,
            streaming=streaming,
        )

    def iter_turn_stream(
        self,
        user_message: str,
    ) -> Iterator[ChatMessage]:
        """Run a streaming turn and yield ``ChatMessage`` events."""
        collected: list[ChatMessage] = []

        def _collect(msg: ChatMessage) -> None:
            collected.append(msg)

        prior = self._on_event
        self._on_event = _collect
        if self._bridge is None:
            self._bridge = ConversationEventBridge(
                session_id=self._session_id,
                handler=_collect,
            )
            self._bridge.register()
        try:
            self.handle_turn(user_message, stream=True)
        finally:
            self._on_event = prior
        yield from collected

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.unregister()
            self._bridge = None
        if self._flow._should_defer_trace_finalization():
            self._flow.finalize_session_traces()

    def _emit(self, msg_type: ChatMessageType, payload: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        self._seq += 1
        self._on_event(
            ChatMessage(
                type=msg_type,
                session_id=self._session_id,
                payload=payload,
                seq=self._seq,
            )
        )


class ConversationEventBridge:
    """Maps CrewAI bus events to ``ChatMessage`` for a session."""

    def __init__(
        self,
        session_id: str,
        handler: Callable[[ChatMessage], None],
    ) -> None:
        self._session_id = session_id
        self._handler = handler
        self._seq = 0
        self._handlers: list[Any] = []

    def register(self) -> None:
        from crewai.events import crewai_event_bus
        from crewai.events.types.flow_events import FlowFinishedEvent
        from crewai.events.types.llm_events import (
            LLMStreamChunkEvent,
            LLMThinkingChunkEvent,
        )
        from crewai.events.types.tool_usage_events import (
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        bus = crewai_event_bus

        @bus.on(LLMStreamChunkEvent)
        def _on_chunk(_source: Any, event: LLMStreamChunkEvent) -> None:
            if not self._matches(event):
                return
            chunk = getattr(event, "chunk", None)
            if chunk:
                self._dispatch(
                    "assistant_delta",
                    {"chunk": chunk, "agent_role": getattr(event, "agent_role", "")},
                )

        @bus.on(LLMThinkingChunkEvent)
        def _on_thinking(_source: Any, event: LLMThinkingChunkEvent) -> None:
            if not self._matches(event):
                return
            chunk = getattr(event, "chunk", None)
            if chunk:
                self._dispatch(
                    "assistant_delta",
                    {
                        "chunk": chunk,
                        "thinking": True,
                        "agent_role": getattr(event, "agent_role", ""),
                    },
                )

        @bus.on(ToolUsageStartedEvent)
        def _on_tool_start(_source: Any, event: ToolUsageStartedEvent) -> None:
            if not self._matches(event):
                return
            self._dispatch(
                "tool_started",
                {"tool_name": getattr(event, "tool_name", "")},
            )

        @bus.on(ToolUsageFinishedEvent)
        def _on_tool_end(_source: Any, event: ToolUsageFinishedEvent) -> None:
            if not self._matches(event):
                return
            self._dispatch(
                "tool_finished",
                {"tool_name": getattr(event, "tool_name", "")},
            )

        @bus.on(FlowFinishedEvent)
        def _on_finished(_source: Any, event: FlowFinishedEvent) -> None:
            if not self._matches(event):
                return
            self._dispatch("turn_finished", {"result": getattr(event, "result", None)})

        self._handlers = [
            _on_chunk,
            _on_thinking,
            _on_tool_start,
            _on_tool_end,
            _on_finished,
        ]

    def unregister(self) -> None:
        self._handlers.clear()

    def _matches(self, event: Any) -> bool:
        meta = getattr(event, "fingerprint_metadata", None) or {}
        if isinstance(meta, dict) and meta.get("conversation_id") == self._session_id:
            return True
        fp = getattr(event, "source_fingerprint", None)
        return fp == self._session_id

    def _dispatch(self, msg_type: ChatMessageType, payload: dict[str, Any]) -> None:
        self._seq += 1
        self._handler(
            ChatMessage(
                type=msg_type,
                session_id=self._session_id,
                payload=payload,
                seq=self._seq,
            )
        )


def stamp_conversation_fingerprint(event: Any, session_id: str) -> None:
    """Stamp ``conversation_id`` on an event before dispatch to external systems."""
    if not getattr(event, "source_fingerprint", None):
        event.source_fingerprint = session_id
    meta = getattr(event, "fingerprint_metadata", None)
    if meta is None:
        event.fingerprint_metadata = {"conversation_id": session_id}
    elif isinstance(meta, dict):
        meta.setdefault("conversation_id", session_id)
