"""Core data models for the NewAgent system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """An artifact attached to a message (file, image, structured data, etc.)."""

    type: str  # "file" | "image" | "json" | "code" | "url"
    name: str = ""
    content: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageAction(BaseModel):
    """A structured action attached to a message.

    Plain-text providers (CLI) ignore these — the user responds
    conversationally. Rich providers (Slack, Teams, Web) render them
    as buttons, cards, or interactive components.
    """

    action_id: str
    label: str
    action_type: str  # "suggestion_confirm" | "suggestion_reject" | "suggestion_edit"
    payload: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single message in a conversation."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    conversation_id: str = ""
    role: str  # "user" | "agent" | "coworker" | "system"
    content: str
    sender: str | None = None
    artifacts: list[Artifact] | None = None
    actions: list[MessageAction] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    response_time_ms: int | None = None

    tools_used: list[str] | None = None
    delegations: list[str] | None = None
    metadata: dict[str, Any] | None = None


class AgentSettings(BaseModel):
    """Opinionated agent settings with sensible defaults."""

    memory_enabled: bool = True
    memory_read_only: bool = False
    reasoning_enabled: bool = True
    self_improving: bool = True

    dreaming_interval_hours: int = 24
    dreaming_trigger_threshold: int = 10
    dreaming_llm: str | Any | None = None

    planning_enabled: bool = True
    auto_plan: bool = True

    can_spawn_copies: bool = False
    max_spawn_depth: int = 1
    max_concurrent_spawns: int = 4
    spawn_timeout: int = 600
    can_create_knowledge: bool = True
    can_build_skills: bool = True
    can_schedule: bool = False

    provenance_enabled: bool = True
    provenance_detail: str = "standard"

    share_data: bool = False

    narration_guard: bool = False
    narration_max_retries: int = 2

    respect_context_window: bool = True
    cache_tool_results: bool = True
    max_retry_limit: int = 2
    max_history_messages: int | None = None


class AgentStatus(BaseModel):
    """Ephemeral status update emitted while the agent works."""

    state: str  # "thinking" | "using_tool" | "delegating" | "planning" | "recalling" | "dreaming"
    detail: str | None = None
    tool_name: str | None = None
    coworker: str | None = None
    progress: float | None = None
    elapsed_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class PromptLayer(BaseModel):
    """A single layer in the prompt stack."""

    name: str
    content: str
    source: str = ""


class PromptStack(BaseModel):
    """Structured system prompt assembly."""

    layers: list[PromptLayer] = Field(default_factory=list)

    def assemble(self) -> str:
        return "\n\n".join(
            layer.content for layer in self.layers if layer.content
        )

    def add(self, name: str, content: str, source: str = "") -> None:
        self.layers.append(PromptLayer(name=name, content=content, source=source))


class ProvenanceEntry(BaseModel):
    """A single decision trace entry."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: str = ""
    action: str  # "tool_call" | "delegation" | "response" | "knowledge_query"
    reasoning: str = ""
    inputs: dict[str, Any] | None = None
    outcome: str | None = None
    confidence: float | None = None
    sources: list[str] | None = None


class TokenUsage(BaseModel):
    """Token consumption record for a single action."""

    action: str  # "message" | "delegation" | "tool_call" | "dreaming" | "planning" | "guardrail"
    agent_id: str = ""
    conversation_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delegation_target: str | None = None
    tool_name: str | None = None
    coworker_source: str | None = None


# ── GAP-45: Memory scoping types ────────────────────────────────


class MemoryScope(BaseModel):
    """Scoped memory namespace."""

    namespace: str
    shared: bool = False  # If True, readable by coworkers


class MemorySlice(BaseModel):
    """Filtered view of memory."""

    scope: str = ""
    user_id: str | None = None
    conversation_id: str | None = None
    tags: list[str] = Field(default_factory=list)
