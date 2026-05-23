from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    register_before_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.hooks.types import BeforeToolCallHookCallable, BeforeToolCallHookType
from crewai.utilities.string_utils import sanitize_tool_name


@dataclass(frozen=True)
class GuardrailRequest:
    """Normalized context passed to a guardrail provider for each tool call."""

    tool_name: str
    tool_input: dict[str, Any]
    agent_role: str | None = None
    task_description: str | None = None
    crew_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class GuardrailDecision:
    """Provider verdict for a tool call."""

    allow: bool
    reason: str | None = None


@runtime_checkable
class GuardrailProvider(Protocol):
    """Protocol for pluggable pre-tool-call authorization providers."""

    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate whether a tool call should proceed."""
        ...


class AllowlistGuardrailProvider:
    """Minimal built-in provider that allows or denies tools by name."""

    name = "allowlist"

    def __init__(
        self,
        *,
        allowed_tools: Iterable[str] | None = None,
        denied_tools: Iterable[str] | None = None,
    ) -> None:
        self._allowed = (
            {sanitize_tool_name(tool_name) for tool_name in allowed_tools}
            if allowed_tools is not None
            else None
        )
        self._denied = {
            sanitize_tool_name(tool_name) for tool_name in (denied_tools or [])
        }

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate a tool call against allowlist and denylist rules."""
        tool_name = request.tool_name

        if tool_name in self._denied:
            return GuardrailDecision(
                allow=False,
                reason=f"'{tool_name}' is in the denied tools list.",
            )

        if self._allowed is not None and tool_name not in self._allowed:
            return GuardrailDecision(
                allow=False,
                reason=f"'{tool_name}' is not in the allowed tools list.",
            )

        return GuardrailDecision(allow=True)


def build_guardrail_request(context: ToolCallHookContext) -> GuardrailRequest:
    """Build a normalized, read-only request snapshot from a tool hook context."""
    crew_id = getattr(context.crew, "id", None)
    return GuardrailRequest(
        tool_name=context.tool_name,
        tool_input=dict(context.tool_input),
        agent_role=getattr(context.agent, "role", None),
        task_description=getattr(context.task, "description", None),
        crew_id=str(crew_id) if crew_id is not None else None,
    )


def enable_guardrail(
    provider: GuardrailProvider,
    *,
    fail_closed: bool = True,
) -> BeforeToolCallHookType | BeforeToolCallHookCallable:
    """Register a guardrail provider as a before_tool_call hook.

    This is strictly additive. It reuses CrewAI's existing hook system and does
    not alter the tool execution pipeline unless the caller explicitly enables
    it. CrewAI before_tool_call hooks currently support allow or block semantics,
    so escalation/delegation workflows should be handled by provider packages on
    top of this seam rather than in core.

    Args:
        provider: Provider used to authorize tool calls.
        fail_closed: When True, provider failures block execution. When False,
            provider failures fall back to allow.

    Returns:
        The registered hook so callers can unregister it later.
    """

    def _hook(context: ToolCallHookContext) -> bool | None:
        try:
            decision = provider.evaluate(build_guardrail_request(context))
        except Exception:
            return False if fail_closed else None

        if decision.allow:
            return None
        return False

    register_before_tool_call_hook(_hook)
    return _hook


def disable_guardrail(
    hook: BeforeToolCallHookType | BeforeToolCallHookCallable,
) -> bool:
    """Unregister a hook that was previously created by enable_guardrail()."""
    return unregister_before_tool_call_hook(hook)
