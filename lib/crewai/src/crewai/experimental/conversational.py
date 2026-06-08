"""Conversational types and helpers shared by ``Flow`` (experimental).

The conversational chat surface (``Flow`` with ``conversational = True``) is
EXPERIMENTAL. APIs in this module and the conversational methods on ``Flow``
may change without a major-version bump until the feature graduates.

This module hosts the **data shapes** — ``ConversationConfig``,
``RouterConfig``, ``ConversationState`` and its message types — plus the
``_conversational_only`` decorator used to gate built-in conversational
methods on the base ``Flow`` class. The methods themselves live on ``Flow``
directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from crewai.utilities.types import LLMMessage


ConversationMessageRole = Literal["user", "assistant", "system", "tool"]
ConversationEventVisibility = Literal["private", "public"]

F = TypeVar("F", bound=Callable[..., Any])


def _conversational_only(func: F) -> F:
    """Mark a method as part of the conversational built-in graph.

    Methods carrying this marker only register on a ``Flow`` subclass when
    ``conversational = True``. Subclasses that don't opt in see them as
    inert attributes — they don't fire and don't pollute the listener graph.
    """
    func.__conversational_only__ = True  # type: ignore[attr-defined]
    return func


@dataclass
class RouterConfig:
    """LLM router configuration for the experimental conversational ``Flow``.

    .. warning::

       **EXPERIMENTAL.** Part of the conversational ``Flow`` surface. Fields
       and defaults may change before the feature graduates from
       ``crewai.experimental``. Pin your CrewAI version if you depend on
       a specific shape.

    ``route_descriptions`` overrides the per-route descriptions used to build
    the router LLM's "available routes" catalog. Routes without an entry fall
    back to the handler's docstring first line (or, for built-in routes, the
    framework's canned description). ``route_permissions`` maps protected route
    labels to one or more permission names; alternatively, pass
    ``required_permissions`` to ``@listen(...)``. Denied turns redirect to
    ``permission_denied_route``. ``prompt`` is reserved for domain policy/voice,
    not the route catalog — that's auto-built.
    """

    prompt: str | None = None
    response_format: type[BaseModel] | None = None
    llm: Any | None = None
    routes: Sequence[str] | None = None
    route_descriptions: dict[str, str] | None = None
    route_permissions: dict[str, str | Sequence[str]] | None = None
    permission_denied_route: str = "permission_denied"
    default_intent: str | None = "converse"
    fallback_intent: str | None = "converse"
    intent_field: str = "intent"


@dataclass
class ConversationConfig:
    """Class-level configuration for the experimental conversational ``Flow``.

    .. warning::

       **EXPERIMENTAL.** Part of the conversational ``Flow`` surface. Fields
       and defaults may change before the feature graduates from
       ``crewai.experimental``. Pin your CrewAI version if you depend on
       a specific shape.

    ``system_prompt`` defaults to the ``slices.conversational_system_prompt``
    translation when left as ``None``. Pass an empty string to opt out of any
    system prompt for ``converse_turn``. ``answer_from_history_prompt`` falls
    back to ``slices.conversational_answer_from_history_prompt`` when ``None``.
    """

    system_prompt: str | None = None
    llm: Any | None = None
    router: RouterConfig | None = None
    answer_from_history_prompt: str | None = None
    default_intents: Sequence[str] | None = None
    intent_llm: Any | None = None
    answer_from_history_llm: Any | None = None
    visible_agent_outputs: Sequence[str] | Literal["all"] | None = None
    defer_trace_finalization: bool = True

    def __call__(self, flow_cls: type[Any]) -> type[Any]:
        """Use this config as a class decorator."""
        flow_cls.conversational_config = self
        return flow_cls


class ConversationMessage(BaseModel):
    """Canonical user-facing message shared across conversational turns."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: ConversationMessageRole
    content: str | list[dict[str, Any]] | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    files: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Private per-agent message or scratch result."""

    role: ConversationMessageRole | str = "assistant"
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationEvent(BaseModel):
    """Structured trace/event that is separate from user-visible messages."""

    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None
    visibility: ConversationEventVisibility = "private"


class ConversationState(BaseModel):
    """Structured state for the experimental conversational ``Flow``.

    .. warning::

       **EXPERIMENTAL.** Field shape and defaults may change before the
       conversational ``Flow`` graduates from ``crewai.experimental``.

    ``messages`` is the canonical user-facing history. Agent/tool scratch work
    belongs in ``events`` or ``agent_threads`` unless explicitly made public.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[ConversationMessage] = Field(default_factory=list)
    current_user_message: str | None = None
    last_user_message: str | None = None
    last_intent: str | None = None
    ended: bool = False
    events: list[ConversationEvent] = Field(default_factory=list)
    agent_threads: dict[str, list[AgentMessage]] = Field(default_factory=dict)
    session_ready: bool = False


def message_to_llm_dict(message: Any) -> LLMMessage:
    """Coerce a stored ``ConversationMessage`` (or dict) into an ``LLMMessage``."""
    if isinstance(message, BaseModel):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {"role": "user", "content": str(message)}

    return cast(
        LLMMessage,
        {key: value for key, value in data.items() if key != "metadata"},
    )


__all__ = [
    "AgentMessage",
    "ConversationConfig",
    "ConversationEvent",
    "ConversationEventVisibility",
    "ConversationMessage",
    "ConversationMessageRole",
    "ConversationState",
    "RouterConfig",
    "_conversational_only",
    "message_to_llm_dict",
]
