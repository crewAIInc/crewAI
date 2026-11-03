"""Experimental Flow bridge for conversational standalone Agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

from opentelemetry import baggage
from pydantic import field_validator

from crewai.agent.core import Agent
from crewai.flow import ConversationConfig, ConversationState, Flow, listen, start
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
        name: str | None = project_name
        conversational: ClassVar[bool] = conversation_enabled
        conversational_config: ClassVar[ConversationConfig] = ConversationConfig(
            defer_trace_finalization=False
        )
        _skip_auto_memory = True

        def route_turn(self, context: dict[str, Any]) -> str:
            return "converse"

        @(listen("converse") if conversation_enabled else start())
        async def converse_turn(self) -> Any:
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
