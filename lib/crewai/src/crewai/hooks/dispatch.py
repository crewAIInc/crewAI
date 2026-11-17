"""Generic interception-hook dispatcher.

This module is the single engine behind every CrewAI interception point. A hook
receives a typed context, may mutate it in place and/or return a replacement
payload, and may raise :class:`HookAborted` to stop the intercepted operation
with a reason and source.

The four public hook families (``before/after_llm_call`` and
``before/after_tool_call``) are adapters registered on this dispatcher, so the
legacy dialect (``register_*``/decorators/``return False``) and the new dialect
(``@on(point)`` / ``HookAborted``) share one ordered queue per point.

Design notes:
- Global registration order is preserved; execution-scoped hooks (via
  ``contextvars``) run after global ones, mirroring
  ``events/event_bus.py``'s ``_runtime_state_var`` scoping pattern.
- ``dispatch`` has a no-op fast path (a single dict lookup) when no hooks are
  registered for a point.
- Hooks are synchronous. They may be invoked from async seams, so they must not
  block on heavy I/O (same restriction as the legacy hooks).
- ``HookAborted`` propagates by design. Any other exception raised by a hook is
  swallowed (fail-open) to preserve the framework's protection against a buggy
  user hook.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
import contextvars
from enum import Enum
from functools import wraps
import inspect
import time
from typing import Any

from crewai.utilities.string_utils import sanitize_tool_name


class InterceptionPoint(str, Enum):
    """Interception points wired by this layer.

    New points are introduced alongside the seams that dispatch them, so the
    enum only ever lists points with a live consumer.
    """

    # Execution-level boundaries
    EXECUTION_START = "execution_start"
    INPUT = "input"
    OUTPUT = "output"
    EXECUTION_END = "execution_end"

    # Model / tool boundaries (legacy-compatible)
    PRE_MODEL_CALL = "pre_model_call"
    POST_MODEL_CALL = "post_model_call"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"

    # Step points
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"


EXECUTION_BOUNDARY_POINTS: frozenset[InterceptionPoint] = frozenset(
    {
        InterceptionPoint.EXECUTION_START,
        InterceptionPoint.INPUT,
        InterceptionPoint.OUTPUT,
        InterceptionPoint.EXECUTION_END,
    }
)
"""The points that open and close a run, as opposed to steps within it."""


class HookAborted(Exception):  # noqa: N818 - public contract name from OSS-86
    """Raised by a hook (or a legacy adapter) to abort the intercepted operation.

    Args:
        reason: Human-readable explanation of why the operation was aborted.
        source: Optional identifier of the aborting hook (callable, string, or
            any object). Used for telemetry and failure messages.
    """

    def __init__(self, reason: str, source: Any = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.source = source


HookFn = Callable[[Any], Any]


class HookErrorPolicy(str, Enum):
    """What dispatch does when a hook raises something other than ``HookAborted``.

    ``FAIL_OPEN`` is the historical behaviour: the error is swallowed so a buggy
    user hook cannot break the framework. ``ABORT`` treats the failure as
    :class:`HookAborted`, which is what a hook acting as a security control needs
    - a hook that cannot reach a decision must not let the operation through.
    """

    FAIL_OPEN = "fail_open"
    ABORT = "abort"


_error_policies: dict[HookFn, HookErrorPolicy] = {}


def set_error_policy(hook: HookFn, policy: HookErrorPolicy | str) -> None:
    """Choose what dispatch does when ``hook`` raises something other than ``HookAborted``.

    Args:
        hook: The callable that was or will be registered, or the original
            function of a filtered hook. It has to be hashable.
        policy: ``"fail_open"`` (the default for every hook) or ``"abort"``.
    """
    resolved = HookErrorPolicy(policy)
    try:
        if resolved is HookErrorPolicy.FAIL_OPEN:
            _error_policies.pop(hook, None)
        else:
            _error_policies[hook] = resolved
    except TypeError as error:
        raise ValueError(
            "on_error='abort' needs a hashable hook callable; wrap it in a function."
        ) from error


def _policy_candidates(hook: Any) -> Iterator[Any]:
    """The callables a policy for ``hook`` may have been registered under.

    The object dispatch invokes is rarely the object the caller holds:

    - a filtered hook (``@before_tool_call(tools=...)``, ``@on(..., agents=...)``,
      a ``@CrewBase`` filter) is registered as a wrapper, which
      ``functools.wraps`` links back through ``__wrapped__``;
    - ``@on`` also stashes the registered wrapper on the original, in
      ``_registered_hook``;
    - a ``@CrewBase`` method is registered as a bound method, while the decorator
      returns - and the docs tell callers to pass - the plain function, which a
      bound method exposes as ``__func__``.
    """
    seen: set[int] = set()
    pending = [hook]
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        pending.append(getattr(candidate, "__wrapped__", None))
        pending.append(getattr(candidate, "_registered_hook", None))
        pending.append(getattr(candidate, "__func__", None))


def _error_policy(hook: HookFn) -> HookErrorPolicy:
    """The error policy registered for ``hook``, fail-open by default."""
    for candidate in _policy_candidates(hook):
        try:
            policy = _error_policies.get(candidate)
        except TypeError:  # an unhashable callable can never carry a policy
            continue
        if policy is not None:
            return policy
    return HookErrorPolicy.FAIL_OPEN


def _pop_error_policy(hook: Any) -> None:
    """Drop the policy of ``hook`` and of the functions it wraps or came from."""
    for candidate in _policy_candidates(hook):
        if isinstance(candidate, Hashable):
            _error_policies.pop(candidate, None)


def clear_error_policy(hook: HookFn) -> None:
    """Public counterpart of :func:`_pop_error_policy`, for registries that
    remove a hook without going through :func:`unregister`."""
    _pop_error_policy(hook)


# (ctx, result) -> modified?  A reducer maps a hook's return value onto the
# context using point-specific semantics. It may raise HookAborted.
Reducer = Callable[[Any, Any], bool]


_global_hooks: dict[InterceptionPoint, list[HookFn]] = {
    point: [] for point in InterceptionPoint
}

_scoped_hooks_var: contextvars.ContextVar[
    dict[InterceptionPoint, list[HookFn]] | None
] = contextvars.ContextVar("crewai_scoped_hooks", default=None)


_TELEMETRY_SOURCE = object()


def get_global_hook_list(point: InterceptionPoint) -> list[HookFn]:
    """Return the live global hook list for a point.

    The returned list object is stable for the lifetime of the process, which
    lets legacy modules alias their module-level registries to it. Mutate it in
    place (append/remove/clear); never rebind it.
    """
    return _global_hooks[point]


def register(point: InterceptionPoint, hook: HookFn) -> None:
    """Register a global hook for an interception point."""
    _global_hooks[point].append(hook)


def unregister(point: InterceptionPoint, hook: HookFn) -> bool:
    """Unregister a specific global hook. Returns True if it was removed.

    When ``hook`` was registered through :func:`on` with ``agents``/``tools``
    filters, the stored callable is a wrapper rather than ``hook`` itself. The
    wrapper is stashed on ``hook._registered_hook`` at registration time, so it
    can be resolved and removed here.
    """
    hooks = _global_hooks[point]
    target = hook if hook in hooks else getattr(hook, "_registered_hook", hook)
    try:
        hooks.remove(target)
        _pop_error_policy(target)
        _pop_error_policy(hook)
        return True
    except ValueError:
        return False


def get_hooks(point: InterceptionPoint) -> list[HookFn]:
    """Return a copy of the global hooks registered for a point."""
    return _global_hooks[point].copy()


def clear(point: InterceptionPoint) -> int:
    """Clear all global hooks for a point. Returns the number cleared."""
    hooks = _global_hooks[point]
    count = len(hooks)
    for hook in hooks:
        _pop_error_policy(hook)
    hooks.clear()
    return count


def clear_all() -> None:
    """Clear all global hooks across every interception point."""
    for hooks in _global_hooks.values():
        for hook in hooks:
            _pop_error_policy(hook)
        hooks.clear()


@contextmanager
def scoped_hooks(
    hooks: dict[InterceptionPoint, list[HookFn]] | None = None,
) -> Iterator[dict[InterceptionPoint, list[HookFn]]]:
    """Enter an execution-scoped hook registry.

    Hooks registered inside this context (via :func:`register_scoped`) run after
    global hooks and are discarded when the context exits. Mirrors the event
    bus's scoped-handler pattern.
    """
    scope: dict[InterceptionPoint, list[HookFn]] = hooks if hooks is not None else {}
    token = _scoped_hooks_var.set(scope)
    try:
        yield scope
    finally:
        _scoped_hooks_var.reset(token)


def register_scoped(point: InterceptionPoint, hook: HookFn) -> None:
    """Register a hook scoped to the current :func:`scoped_hooks` context."""
    scope = _scoped_hooks_var.get()
    if scope is None:
        raise RuntimeError(
            "register_scoped() called outside of a scoped_hooks() context"
        )
    scope.setdefault(point, []).append(hook)


def get_scoped_hooks(point: InterceptionPoint) -> list[HookFn]:
    """Return the hooks registered in the current execution scope for a point.

    Used by seams that carry a pre-snapshotted hook list (e.g. the agent
    executors' per-executor LLM hook lists) so they can merge in
    execution-scoped hooks with the same snapshot-then-scoped ordering that
    :func:`dispatch` applies to global vs scoped hooks.
    """
    scope = _scoped_hooks_var.get()
    if not scope:
        return []
    return list(scope.get(point, []))


def _resolve_hooks(point: InterceptionPoint) -> list[HookFn]:
    """Resolve the ordered hooks for a point: global first, then scoped."""
    global_hooks = _global_hooks[point]
    scope = _scoped_hooks_var.get()
    if scope:
        scoped = scope.get(point)
        if scoped:
            return [*global_hooks, *scoped]
    return global_hooks


def source_name(source: Any) -> str | None:
    """Best-effort readable name for a hook source."""
    if source is None:
        return None
    if isinstance(source, str):
        return source
    name = getattr(source, "__name__", None)
    if isinstance(name, str):
        return name
    return type(source).__name__


def _emit_telemetry(
    point: InterceptionPoint,
    outcome: str,
    hook_count: int,
    duration_ms: float,
    abort_reason: str | None,
    abort_source: str | None,
) -> None:
    """Emit a HookDispatchedEvent. Never raises."""
    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.hook_events import HookDispatchedEvent

        crewai_event_bus.emit(
            _TELEMETRY_SOURCE,
            event=HookDispatchedEvent(
                interception_point=point.value,
                outcome=outcome,
                hook_count=hook_count,
                duration_ms=duration_ms,
                abort_reason=abort_reason,
                abort_source=abort_source,
            ),
        )
    except Exception:  # noqa: S110 - telemetry must never break dispatch
        pass


def _default_reducer(ctx: Any, result: Any) -> bool:
    """Default payload semantics: a non-None return replaces ``ctx.payload``.

    Only reports a modification when the payload was actually applied, so a
    context without a ``payload`` attribute does not produce a misleading
    ``"modified"`` telemetry outcome.
    """
    if result is not None and hasattr(ctx, "payload"):
        ctx.payload = result
        return True
    return False


def _invoke_hook(
    point: InterceptionPoint,
    hook: HookFn,
    ctx: Any,
    reducer: Reducer,
    verbose: bool,
) -> bool:
    """Run a single hook and apply its result via the reducer.

    Returns whether the context was modified. Raises :class:`HookAborted` (with
    ``source`` populated) to abort; any other exception is swallowed (fail-open).
    """
    try:
        result = hook(ctx)
        return reducer(ctx, result)
    except HookAborted as aborted:
        if aborted.source is None:
            aborted.source = hook
        raise
    except Exception as error:
        if _error_policy(hook) is HookErrorPolicy.ABORT:
            raise HookAborted(
                f"{source_name(hook) or 'hook'} failed: {error}", source=hook
            ) from error
        if verbose:
            from crewai_core.printer import PRINTER

            PRINTER.print(
                content=f"Error in {point.value} hook: {error}",
                color="yellow",
            )
        return False


def run_hooks(
    point: InterceptionPoint,
    ctx: Any,
    hooks: list[HookFn],
    *,
    reducer: Reducer | None = None,
    verbose: bool = True,
) -> Any:
    """Execute an explicit list of hooks against a context.

    This is the shared engine used both by :func:`dispatch` (which resolves
    global + scoped hooks) and by seams that carry a pre-snapshotted hook list
    (e.g. per-executor LLM hook lists).

    Args:
        point: The interception point being dispatched.
        ctx: The typed context passed to each hook (mutated in place).
        hooks: The ordered hooks to run.
        reducer: Maps each hook's return value onto ``ctx``. Defaults to
            :func:`_default_reducer` (payload replacement). May raise
            :class:`HookAborted`.
        verbose: Whether to print swallowed-hook-error warnings.

    Returns:
        The (possibly mutated) context.

    Raises:
        HookAborted: If a hook or the reducer aborts the operation. Telemetry is
            still emitted before propagating.
    """
    if not hooks:
        return ctx

    active_reducer = reducer if reducer is not None else _default_reducer
    start = time.perf_counter()
    outcome = "proceeded"
    abort_reason: str | None = None
    abort_source: str | None = None
    modified = False

    try:
        for hook in list(hooks):
            if _invoke_hook(point, hook, ctx, active_reducer, verbose):
                modified = True
        outcome = "modified" if modified else "proceeded"
        return ctx
    except HookAborted as aborted:
        outcome = "aborted"
        abort_reason = aborted.reason
        abort_source = source_name(aborted.source)
        raise
    finally:
        _emit_telemetry(
            point,
            outcome,
            len(hooks),
            (time.perf_counter() - start) * 1000.0,
            abort_reason,
            abort_source,
        )


def dispatch(
    point: InterceptionPoint,
    ctx: Any,
    *,
    reducer: Reducer | None = None,
    verbose: bool = True,
) -> Any:
    """Dispatch a context to all hooks registered for a point.

    Resolves global then scoped hooks and runs them through :func:`run_hooks`.
    No-op fast path when nothing is registered.
    """
    hooks = _resolve_hooks(point)
    if not hooks:
        return ctx
    return run_hooks(point, ctx, hooks, reducer=reducer, verbose=verbose)


def _wrap_with_filters(
    func: HookFn,
    agents: list[str] | None,
    tools: list[str] | None,
) -> HookFn:
    """Wrap a hook so it only runs for matching agents/tools (context-shape aware)."""

    @wraps(func)
    def filtered(ctx: Any) -> Any:
        if tools:
            tool_name = getattr(ctx, "tool_name", None)
            if tool_name is not None and tool_name not in tools:
                return None
        if agents:
            agent = getattr(ctx, "agent", None)
            role = getattr(agent, "role", None) if agent is not None else None
            if role is None:
                role = getattr(ctx, "agent_role", None)
            if role is not None and role not in agents:
                return None
        return func(ctx)

    return filtered


def on(
    point: InterceptionPoint,
    *,
    agents: list[str] | None = None,
    tools: list[str] | None = None,
) -> Callable[[HookFn], HookFn]:
    """Register a function as a hook for an interception point.

    Mirrors the legacy decorators' ergonomics: supports ``agents=`` / ``tools=``
    filters and, when applied to a method inside a ``@CrewBase`` class, defers
    registration to crew initialization (crew-scoped) instead of registering
    globally.

    Example:
        >>> @on(InterceptionPoint.PRE_TOOL_CALL, tools=["delete_file"])
        ... def guard(ctx):
        ...     raise HookAborted("deletion not allowed")
    """
    normalized_tools = [sanitize_tool_name(t) for t in tools] if tools else None

    def decorator(func: HookFn) -> HookFn:
        func._interception_point = point  # type: ignore[attr-defined]
        if normalized_tools:
            func._filter_tools = normalized_tools  # type: ignore[attr-defined]
        if agents:
            func._filter_agents = agents  # type: ignore[attr-defined]

        params = list(inspect.signature(func).parameters.keys())
        is_method = len(params) >= 2 and params[0] == "self"

        if not is_method:
            hook = (
                _wrap_with_filters(func, agents, normalized_tools)
                if (agents or normalized_tools)
                else func
            )
            register(point, hook)
            # Remember the actually-registered callable so unregister_hook(func)
            # can resolve the filter wrapper.
            func._registered_hook = hook  # type: ignore[attr-defined]

        return func

    return decorator
