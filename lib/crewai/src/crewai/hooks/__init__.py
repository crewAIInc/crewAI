from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Protocol, runtime_checkable

from crewai.hooks.decorators import (
    after_llm_call,
    after_tool_call,
    before_llm_call,
    before_tool_call,
)
from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    clear_after_llm_call_hooks,
    clear_all_llm_call_hooks,
    clear_before_llm_call_hooks,
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
    register_after_llm_call_hook,
    register_before_llm_call_hook,
    unregister_after_llm_call_hook,
    unregister_before_llm_call_hook,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    clear_after_tool_call_hooks,
    clear_all_tool_call_hooks,
    clear_before_tool_call_hooks,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    unregister_after_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.hooks.types import BeforeToolCallHookType


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """Detached context for one provider decision before a tool call."""

    tool_name: str
    tool_alias: str
    tool_input: Mapping[str, Any]
    agent_id: str | None = None
    agent_role: str | None = None
    task_description: str | None = None
    crew_id: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Provider verdict for one proposed tool call."""

    allow: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PolicyProvider(Protocol):
    """Contract for pluggable pre-tool-call policy providers."""

    name: str

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        """Evaluate whether the requested tool call may proceed."""
        ...


def enable_policy_provider(
    provider: PolicyProvider,
    *,
    fail_closed: bool = True,
) -> BeforeToolCallHookType:
    """Register a provider through CrewAI's existing before-tool-call hooks.

    Provider failures and invalid results block execution by default. Set
    ``fail_closed=False`` only when availability is intentionally preferred.
    The returned hook can be removed with ``unregister_before_tool_call_hook``.
    """

    def _hook(context: ToolCallHookContext) -> bool | None:
        try:
            request = _build_policy_request(context)
            decision = provider.evaluate(request)
            if not isinstance(decision, PolicyDecision):
                raise TypeError("PolicyProvider.evaluate() must return PolicyDecision")
            if type(decision.allow) is not bool:
                raise TypeError("PolicyDecision.allow must be a bool")
        except Exception:
            logger.exception("PolicyProvider evaluation failed")
            return False if fail_closed else None

        return None if decision.allow else False

    register_before_tool_call_hook(_hook)
    return _hook


def _build_policy_request(context: ToolCallHookContext) -> PolicyRequest:
    """Snapshot the mutable tool-call context before provider evaluation."""

    original_tool_name = _optional_text(getattr(context, "tool", None), "name")
    return PolicyRequest(
        tool_name=original_tool_name or context.tool_name,
        tool_alias=context.tool_name,
        tool_input=deepcopy(context.tool_input),
        agent_id=_optional_identifier(context.agent, "id"),
        agent_role=_optional_text(context.agent, "role"),
        task_description=_optional_text(context.task, "description"),
        crew_id=_optional_identifier(context.crew, "id"),
    )


def _optional_text(source: Any, attribute: str) -> str | None:
    if source is None:
        return None
    value = getattr(source, attribute, None)
    return value if isinstance(value, str) and value else None


def _optional_identifier(source: Any, attribute: str) -> str | None:
    if source is None:
        return None
    value = getattr(source, attribute, None)
    if value is None:
        return None
    rendered = str(value)
    return rendered if rendered else None


def clear_all_global_hooks() -> dict[str, tuple[int, int]]:
    """Clear all global hooks across all hook types (LLM and Tool).

    This is a convenience function that clears all registered hooks in one call.
    Useful for testing, resetting state, or cleaning up between different
    execution contexts.

    Returns:
        Dictionary with counts of cleared hooks:
        {
            "llm_hooks": (before_count, after_count),
            "tool_hooks": (before_count, after_count),
            "total": (total_before_count, total_after_count)
        }

    Example:
        >>> # Register various hooks
        >>> register_before_llm_call_hook(llm_hook1)
        >>> register_after_llm_call_hook(llm_hook2)
        >>> register_before_tool_call_hook(tool_hook1)
        >>> register_after_tool_call_hook(tool_hook2)
        >>>
        >>> # Clear all hooks at once
        >>> result = clear_all_global_hooks()
        >>> print(result)
        {
            'llm_hooks': (1, 1),
            'tool_hooks': (1, 1),
            'total': (2, 2)
        }
    """
    llm_counts = clear_all_llm_call_hooks()
    tool_counts = clear_all_tool_call_hooks()

    return {
        "llm_hooks": llm_counts,
        "tool_hooks": tool_counts,
        "total": (llm_counts[0] + tool_counts[0], llm_counts[1] + tool_counts[1]),
    }


__all__ = [
    "LLMCallHookContext",
    "PolicyDecision",
    "PolicyProvider",
    "PolicyRequest",
    "ToolCallHookContext",
    "after_llm_call",
    "after_tool_call",
    "before_llm_call",
    "before_tool_call",
    "clear_after_llm_call_hooks",
    "clear_after_tool_call_hooks",
    "clear_all_global_hooks",
    "clear_all_llm_call_hooks",
    "clear_all_tool_call_hooks",
    "clear_before_llm_call_hooks",
    "clear_before_tool_call_hooks",
    "enable_policy_provider",
    "get_after_llm_call_hooks",
    "get_after_tool_call_hooks",
    "get_before_llm_call_hooks",
    "get_before_tool_call_hooks",
    "register_after_llm_call_hook",
    "register_after_tool_call_hook",
    "register_before_llm_call_hook",
    "register_before_tool_call_hook",
    "unregister_after_llm_call_hook",
    "unregister_after_tool_call_hook",
    "unregister_before_llm_call_hook",
    "unregister_before_tool_call_hook",
]
