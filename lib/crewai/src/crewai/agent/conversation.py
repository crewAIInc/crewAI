"""Experimental Flow bridge for conversational standalone Agents."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any, ClassVar, cast

from opentelemetry import baggage
from pydantic import PrivateAttr, field_validator

from crewai.agent.core import Agent
from crewai.flow import ConversationConfig, ConversationState, Flow, listen, start
from crewai.lite_agent_output import LiteAgentOutput
from crewai.utilities.types import LLMMessage


class AgentConversationState(ConversationState):
    @field_validator("messages", mode="before")
    @classmethod
    def normalize_messages(cls, value: Any) -> Any:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("messages must be a non-empty string or message list")
            return [{"role": "user", "content": value}]
        return value


def agent_flow(
    load_agent: Callable[[], Agent],
    *,
    name: str = "Agent",
    conversational: bool = True,
) -> type[Flow[AgentConversationState]]:
    """Build a Flow that calls the Agent loader for each kickoff or chat turn."""

    project_name = name
    conversation_enabled = conversational

    class AgentFlow(Flow[AgentConversationState]):
        _standalone_agent: ClassVar[bool] = True
        _turn_lock: Any = PrivateAttr(default_factory=Lock)
        _bound_session_id: str | None = PrivateAttr(default=None)
        name: str | None = project_name
        conversational: ClassVar[bool] = conversation_enabled
        conversational_config: ClassVar[ConversationConfig] = ConversationConfig(
            defer_trace_finalization=False
        )
        _skip_auto_memory = True

        @contextmanager
        def _conversation_turn_scope(self, session_id: str) -> Iterator[None]:
            if not self._turn_lock.acquire(blocking=False):
                raise RuntimeError("An Agent conversation turn is already running")
            try:
                bound = self._bound_session_id
                if bound is None and self.state.messages:
                    bound = self.state.id
                if bound is not None and session_id != bound:
                    raise ValueError(
                        "An Agent conversation belongs to one session; "
                        "create a new Agent for a different session"
                    )
                self._bound_session_id = session_id
                self.state.id = session_id
                yield
            finally:
                self._turn_lock.release()

        def chat(self, *, session_id: str | None = None, **kwargs: Any) -> None:
            with self._conversation_turn_scope(session_id or self.state.id):
                pass
            super().chat(session_id=session_id, **kwargs)

        def route_turn(self, context: dict[str, Any]) -> str:
            return "converse"

        @(listen("converse") if conversation_enabled else start())
        async def converse_turn(self) -> LiteAgentOutput:
            messages = [
                cast(
                    LLMMessage,
                    message.model_dump(exclude={"metadata"}, exclude_unset=True),
                )
                for message in self.state.messages
            ]
            if not messages:
                raise ValueError("messages must be a non-empty string or message list")
            agent = load_agent()
            with self._conversation_streaming_enabled(agent.llm):
                result = await agent.kickoff_async(
                    messages=messages,
                    input_files=cast(
                        "dict[str, Any] | None",
                        baggage.get_baggage("flow_input_files") or None,
                    ),
                )
            if conversation_enabled:
                self.append_assistant_message(result.raw)
            return result

    return AgentFlow
