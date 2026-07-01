"""
Standalone enable_guardrail implementation.

Used when CrewAI's native enable_guardrail is not available (e.g., before
the upstream PR is merged). Provides the same API surface for testing
and development purposes.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .types import GuardrailDecision, GuardrailProvider, GuardrailRequest


@dataclass
class ToolCallContext:
    """Minimal tool-call context for standalone testing."""

    tool_name: str
    tool_input: dict[str, Any]
    agent_id: str | None = None
    agent_role: str | None = None
    task_description: str | None = None
    crew_id: str | None = None


# Global registry for standalone hooks
_hooks: list[Callable[[ToolCallContext], bool | None]] = []


def enable_guardrail(
    provider: GuardrailProvider,
    *,
    fail_closed: bool = True,
) -> Callable[[ToolCallContext], bool | None]:
    """Register a provider as a global before-tool-call authorization hook.

    Provider and request-construction failures block execution by default.
    Set ``fail_closed=False`` only when availability is more important than
    enforcing authorization.

    Returns the registered hook function.
    """

    def _hook(context: ToolCallContext) -> bool | None:
        try:
            request = _build_request(context)
            decision = provider.evaluate(request)
            if not isinstance(decision, GuardrailDecision):
                raise TypeError(
                    "GuardrailProvider.evaluate() must return GuardrailDecision"
                )
            if type(decision.allow) is not bool:
                raise TypeError("GuardrailDecision.allow must be a bool")
        except Exception:
            return False if fail_closed else None
        return None if decision.allow else False

    _hooks.append(_hook)
    return _hook


def _build_request(context: ToolCallContext) -> GuardrailRequest:
    """Build a GuardrailRequest from a tool-call context."""
    return GuardrailRequest(
        tool_name=context.tool_name,
        tool_input=deepcopy(context.tool_input),
        agent_id=context.agent_id,
        agent_role=context.agent_role,
        task_description=context.task_description,
        crew_id=context.crew_id,
        timestamp=time.time(),
    )


def clear_hooks() -> None:
    """Clear all registered standalone hooks."""
    _hooks.clear()


def get_hooks() -> list[Callable[[ToolCallContext], bool | None]]:
    """Return all registered standalone hooks."""
    return list(_hooks)
