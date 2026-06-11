"""Static conversational Flow definition models.

This module is part of the serializable Flow Definition contract. It should
only contain static data shapes. Experimental conversational runtime behavior
continues to live in ``crewai.experimental.conversational_mixin``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FlowConversationalRouterDefinition(BaseModel):
    """Static conversational router configuration."""

    prompt: str | None = None
    response_format: Any = None
    llm: Any = None
    routes: list[str] | None = None
    route_descriptions: dict[str, str] | None = None
    default_intent: str | None = "converse"
    fallback_intent: str | None = "converse"
    intent_field: str = "intent"


class FlowConversationalDefinition(BaseModel):
    """Static conversational Flow configuration."""

    enabled: bool = False
    system_prompt: str | None = None
    llm: Any = None
    router: FlowConversationalRouterDefinition | None = None
    answer_from_history_prompt: str | None = None
    default_intents: list[str] | None = None
    intent_llm: Any = None
    answer_from_history_llm: Any = None
    visible_agent_outputs: list[str] | Literal["all"] | None = None
    defer_trace_finalization: bool = True
    builtin_routes: list[str] = Field(default_factory=lambda: ["converse", "end"])
    internal_routes: list[str] = Field(default_factory=lambda: ["answer_from_history"])


__all__ = [
    "FlowConversationalDefinition",
    "FlowConversationalRouterDefinition",
]
