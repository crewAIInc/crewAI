"""GuardrailProvider interface for pre-tool-call authorization.

This module provides a standard protocol for pluggable tool-call authorization
that sits on top of CrewAI's existing BeforeToolCallHook system.

Usage:
    Simple provider that blocks specific tools::

        from crewai.hooks import (
            GuardrailProvider,
            GuardrailRequest,
            GuardrailDecision,
            enable_guardrail,
        )

        class BlockListProvider:
            name = "block_list"

            def __init__(self, blocked_tools: list[str]) -> None:
                self.blocked_tools = blocked_tools

            def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
                if request.tool_name in self.blocked_tools:
                    return GuardrailDecision(
                        allow=False,
                        reason=f"Tool '{request.tool_name}' is blocked by policy",
                    )
                return GuardrailDecision(allow=True)

            def health_check(self) -> bool:
                return True

        provider = BlockListProvider(blocked_tools=["ShellTool", "dangerous_op"])
        disable = enable_guardrail(provider)

        # Later, to remove the guardrail:
        disable()

    Rate-limiting provider::

        import time
        from collections import defaultdict

        class RateLimitProvider:
            name = "rate_limiter"

            def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
                self.max_calls = max_calls
                self.window_seconds = window_seconds
                self._calls: dict[str, list[float]] = defaultdict(list)

            def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
                now = time.time()
                key = request.tool_name
                # Remove expired entries
                self._calls[key] = [
                    t for t in self._calls[key]
                    if now - t < self.window_seconds
                ]
                if len(self._calls[key]) >= self.max_calls:
                    return GuardrailDecision(
                        allow=False,
                        reason=f"Rate limit exceeded for '{key}'",
                    )
                self._calls[key].append(now)
                return GuardrailDecision(allow=True)

            def health_check(self) -> bool:
                return True

    Per-agent role restriction::

        class RoleBasedProvider:
            name = "role_based"

            def __init__(self, permissions: dict[str, list[str]]) -> None:
                # Maps agent role -> list of allowed tool names
                self.permissions = permissions

            def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
                role = request.agent_role
                if role is None:
                    return GuardrailDecision(allow=True)
                allowed = self.permissions.get(role)
                if allowed is not None and request.tool_name not in allowed:
                    return GuardrailDecision(
                        allow=False,
                        reason=(
                            f"Agent '{role}' is not permitted "
                            f"to use '{request.tool_name}'"
                        ),
                    )
                return GuardrailDecision(allow=True)

            def health_check(self) -> bool:
                return True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import datetime
import logging
from typing import Protocol, runtime_checkable

from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    register_before_tool_call_hook,
    unregister_before_tool_call_hook,
)


logger = logging.getLogger(__name__)


@dataclass
class GuardrailRequest:
    """Context passed to the provider for each tool call.

    Attributes:
        tool_name: Name of the tool being invoked.
        tool_input: Dictionary of arguments passed to the tool.
        agent_role: Role of the agent executing the tool (may be ``None``).
        task_description: Description of the current task (may be ``None``).
        crew_id: Identifier for the crew instance (may be ``None``).
        timestamp: ISO 8601 timestamp of when the request was created.
    """

    tool_name: str
    tool_input: dict[str, object]
    agent_role: str | None = None
    task_description: str | None = None
    crew_id: str | None = None
    timestamp: str = ""


@dataclass
class GuardrailDecision:
    """Provider's allow/deny verdict for a tool call.

    Attributes:
        allow: ``True`` to permit execution, ``False`` to block it.
        reason: Human-readable explanation (surfaced to the agent when blocked).
        metadata: Arbitrary provider-specific data (e.g. policy ID, audit ref).
    """

    allow: bool
    reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class GuardrailProvider(Protocol):
    """Contract for pluggable tool-call authorization.

    Any class that implements this protocol can be wired into CrewAI's
    hook system via :func:`enable_guardrail` to authorize or deny
    individual tool calls before they execute.

    Attributes:
        name: Short identifier for logging / audit purposes.

    Example:
        >>> class MyProvider:
        ...     name = "my_provider"
        ...
        ...     def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        ...         if request.tool_name == "dangerous_tool":
        ...             return GuardrailDecision(allow=False, reason="Blocked")
        ...         return GuardrailDecision(allow=True)
        ...
        ...     def health_check(self) -> bool:
        ...         return True
    """

    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate whether a tool call should proceed.

        Args:
            request: Context about the pending tool call.

        Returns:
            A :class:`GuardrailDecision`. If ``allow`` is ``False``, the tool
            call is blocked and ``reason`` is surfaced to the agent.
        """
        ...

    def health_check(self) -> bool:
        """Optional readiness probe.

        Returns:
            ``True`` if the provider is healthy and ready, ``False`` otherwise.
            The default expectation is ``True``.
        """
        ...


def _build_guardrail_request(context: ToolCallHookContext) -> GuardrailRequest:
    """Build a :class:`GuardrailRequest` from a :class:`ToolCallHookContext`.

    Args:
        context: The hook context for the current tool call.

    Returns:
        A populated :class:`GuardrailRequest`.
    """
    agent_role: str | None = None
    if context.agent is not None and hasattr(context.agent, "role"):
        agent_role = context.agent.role

    task_description: str | None = None
    if context.task is not None and hasattr(context.task, "description"):
        task_description = context.task.description

    crew_id: str | None = None
    if context.crew is not None and hasattr(context.crew, "id"):
        crew_id = str(context.crew.id)

    return GuardrailRequest(
        tool_name=context.tool_name,
        tool_input=context.tool_input,
        agent_role=agent_role,
        task_description=task_description,
        crew_id=crew_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def enable_guardrail(
    provider: GuardrailProvider,
    *,
    fail_closed: bool = True,
) -> Callable[[], bool]:
    """Wire a :class:`GuardrailProvider` into CrewAI's hook system.

    This registers a ``BeforeToolCallHook`` that delegates authorization
    decisions to the given *provider*.  The returned callable can be used
    to remove the hook later.

    Args:
        provider: An object satisfying the :class:`GuardrailProvider` protocol.
        fail_closed: When ``True`` (the default), any exception raised by
            ``provider.evaluate()`` causes the tool call to be **blocked**.
            When ``False``, exceptions are logged and the tool call is
            **allowed** to proceed.

    Returns:
        A ``disable`` callable.  Calling ``disable()`` unregisters the
        hook and returns ``True`` if it was still registered, ``False``
        otherwise.

    Example:
        >>> disable = enable_guardrail(my_provider, fail_closed=True)
        >>> # ... run crews / agents ...
        >>> disable()  # remove the guardrail
        True
    """

    def _hook(context: ToolCallHookContext) -> bool | None:
        request = _build_guardrail_request(context)
        try:
            decision = provider.evaluate(request)
        except Exception:
            logger.exception(
                "GuardrailProvider '%s' raised an exception (fail_closed=%s)",
                provider.name,
                fail_closed,
            )
            return False if fail_closed else None

        if not decision.allow:
            logger.info(
                "GuardrailProvider '%s' denied tool '%s': %s",
                provider.name,
                context.tool_name,
                decision.reason,
            )
            return False  # block tool execution

        return None  # allow tool execution

    register_before_tool_call_hook(_hook)

    def disable() -> bool:
        """Unregister the guardrail hook.

        Returns:
            ``True`` if the hook was found and removed, ``False`` otherwise.
        """
        return unregister_before_tool_call_hook(_hook)

    return disable
