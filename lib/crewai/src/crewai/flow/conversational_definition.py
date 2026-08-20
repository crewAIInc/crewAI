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
    llm: Any = Field(
        default=None,
        description=(
            "Model for the built-in converse handler, and the router's fallback. "
            "A model-id string, or a config mapping such as "
            "{model, max_tokens} -- the same shapes a crew agent's llm accepts."
        ),
        examples=["gpt-4o-mini", {"model": "openai/gpt-4o-mini", "max_tokens": 4096}],
    )
    routes: list[str] | None = None
    route_descriptions: dict[str, str] | None = None
    default_intent: str | None = "converse"
    fallback_intent: str | None = "converse"
    intent_field: str = "intent"


class FlowConversationalDefinition(BaseModel):
    """Static conversational Flow configuration.

    The block is absent (``FlowDefinition.conversational is None``) on flows
    that are not conversational, so declaring it at all is the opt-in.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Whether conversational mode is active. Declaring the conversational "
            "block is the opt-in, so this defaults to true; set it to false to "
            "keep the configuration while turning chat off."
        ),
        examples=[True],
    )
    system_prompt: str | None = None
    llm: Any = None
    router: FlowConversationalRouterDefinition | None = None
    answer_from_history_prompt: str | None = None
    default_intents: list[str] | None = None
    intent_llm: Any = Field(
        default=None,
        description="Model used to pre-classify default_intents. Same shapes as llm.",
        examples=["gpt-4o-mini"],
    )
    answer_from_history_llm: Any = Field(
        default=None,
        description=(
            "Setting this enables the optional answer_from_history route. "
            "Same shapes as llm."
        ),
        examples=["gpt-4o-mini"],
    )
    visible_agent_outputs: list[str] | Literal["all"] | None = None
    defer_trace_finalization: bool = True
    builtin_routes: list[str] = Field(default_factory=lambda: ["converse", "end"])
    internal_routes: list[str] = Field(default_factory=lambda: ["answer_from_history"])


__all__ = [
    "FlowConversationalDefinition",
    "FlowConversationalRouterDefinition",
]
