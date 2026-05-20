"""Conversational turn helpers for CrewAI Flows.

Provides message history utilities, kickoff input normalization, and optional
class-level defaults via ``ConversationalConfig``. Session identity is ``state.id``
(``inputs["id"]`` / ``kickoff(session_id=...)``), not a separate Flow field.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.flow.flow import Flow
    from crewai.llms.base_llm import BaseLLM


TurnMode = Literal["auto", "follow_up", "initial"]

_EXIT_COMMANDS_DEFAULT: tuple[str, ...] = ("exit", "quit")


class ConversationalInputs(TypedDict, total=False):
    """Conventional ``kickoff(inputs=...)`` keys for chat turns."""

    id: str
    user_message: str | dict[str, Any]
    last_intent: str


@dataclass
class ConversationalConfig:
    """Optional class-level defaults for conversational flows.

    Override per kickoff via ``user_message``, ``session_id``, ``intents``, etc.
    """

    default_intents: Sequence[str] | None = None
    intent_llm: str | None = None
    interactive_prompt: str = "You: "
    interactive_timeout: float | None = None
    exit_commands: Sequence[str] = field(default_factory=lambda: _EXIT_COMMANDS_DEFAULT)
    defer_trace_finalization: bool = True


class ChatState(BaseModel):
    """Recommended persisted state shape for multi-turn flows."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[LLMMessage] = Field(default_factory=list)
    last_user_message: str | None = None
    last_intent: str | None = None
    session_ready: bool = False


def _coerce_user_message_text(user_message: str | dict[str, Any] | Any) -> str:
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, dict):
        content = user_message.get("content")
        if content is not None:
            return str(content)
    return str(user_message)


def normalize_kickoff_inputs(
    inputs: dict[str, Any] | None,
    *,
    user_message: str | dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Merge conversational kickoff kwargs into the inputs dict."""
    merged: dict[str, Any] = dict(inputs or {})

    if session_id is not None:
        merged["id"] = session_id

    if user_message is not None:
        merged["user_message"] = user_message
    elif "user_message" in merged and isinstance(merged["user_message"], str):
        pass

    return merged


def get_conversation_messages(flow: Flow[Any]) -> list[LLMMessage]:
    """Read message history from flow state or the internal fallback buffer."""
    buffer: list[LLMMessage] = getattr(flow, "_conversation_messages", [])
    state = getattr(flow, "_state", None)
    if state is None:
        return list(buffer)

    if isinstance(state, dict):
        messages = state.get("messages")
        if isinstance(messages, list):
            return cast(list[LLMMessage], messages)
    elif isinstance(state, BaseModel) and hasattr(state, "messages"):
        messages = getattr(state, "messages", None)
        if isinstance(messages, list):
            return cast(list[LLMMessage], messages)

    return list(buffer)


def append_message(
    flow: Flow[Any],
    role: Literal["user", "assistant", "system", "tool"],
    content: str,
    **extra: Any,
) -> None:
    """Append a message to ``state.messages`` or the flow fallback buffer."""
    message: LLMMessage = {"role": role, "content": content}
    for key, value in extra.items():
        if key in ("tool_call_id", "name", "tool_calls", "files"):
            message[key] = value  # type: ignore[literal-required]

    state = getattr(flow, "_state", None)
    if state is not None:
        if isinstance(state, dict):
            messages = state.get("messages")
            if isinstance(messages, list):
                messages.append(message)
                return
        elif isinstance(state, BaseModel) and hasattr(state, "messages"):
            messages = getattr(state, "messages", None)
            if messages is None:
                object.__setattr__(state, "messages", [])
                messages = state.messages
            if isinstance(messages, list):
                messages.append(message)
                return

    if not hasattr(flow, "_conversation_messages"):
        object.__setattr__(flow, "_conversation_messages", [])
    flow._conversation_messages.append(message)


def set_state_field(flow: Flow[Any], name: str, value: Any) -> None:
    """Set a field on structured or dict flow state when present."""
    state = getattr(flow, "_state", None)
    if state is None:
        return
    if isinstance(state, dict):
        state[name] = value
    elif isinstance(state, BaseModel) and hasattr(state, name):
        object.__setattr__(state, name, value)


def receive_user_message(
    flow: Flow[Any],
    text: str,
    *,
    outcomes: Sequence[str] | None = None,
    llm: str | BaseLLM | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Record a user turn: append message and optionally classify intent."""
    append_message(flow, "user", text)
    set_state_field(flow, "last_user_message", text)

    if outcomes and llm is not None:
        intent = flow.classify_intent(
            text,
            outcomes,
            llm=llm,
            context=get_conversation_messages(flow),
        )
        set_state_field(flow, "last_intent", intent)
        return intent

    return text


def prepare_conversational_turn(
    flow: Flow[Any],
    *,
    user_message: str | dict[str, Any] | None = None,
    intents: Sequence[str] | None = None,
    intent_llm: str | BaseLLM | None = None,
    config: ConversationalConfig | None = None,
) -> None:
    """Hydrate conversation state after inputs are merged into flow state."""
    if user_message is None:
        state = getattr(flow, "_state", None)
        if isinstance(state, dict) and "user_message" in state:
            user_message = state["user_message"]
        elif isinstance(state, BaseModel) and hasattr(state, "user_message"):
            user_message = getattr(state, "user_message", None)

    if user_message is None:
        return

    text = _coerce_user_message_text(user_message)
    if not text.strip():
        return

    # Fresh classification each turn (do not reuse prior turn's route label).
    set_state_field(flow, "last_intent", None)

    resolved_intents = intents
    if resolved_intents is None and config is not None:
        resolved_intents = config.default_intents

    resolved_llm = intent_llm
    if resolved_llm is None and config is not None:
        resolved_llm = config.intent_llm

    if resolved_intents:
        if resolved_llm is None:
            raise ValueError("intent_llm is required when intents are provided")
        receive_user_message(
            flow,
            text,
            outcomes=resolved_intents,
            llm=resolved_llm,
        )
    else:
        receive_user_message(flow, text)


def input_history_to_messages(entries: Sequence[Any]) -> list[LLMMessage]:
    """Convert ``Flow.input_history`` entries to LLM message format."""
    messages: list[LLMMessage] = []
    for entry in entries:
        prompt = entry.get("message") if isinstance(entry, dict) else None
        response = entry.get("response") if isinstance(entry, dict) else None
        if prompt:
            messages.append({"role": "assistant", "content": str(prompt)})
        if response:
            messages.append({"role": "user", "content": str(response)})
    return messages


def get_conversational_config(flow: Flow[Any]) -> ConversationalConfig | None:
    """Return class-level ``conversational_config`` if defined."""
    return getattr(type(flow), "conversational_config", None)
