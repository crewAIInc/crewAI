"""crewAI's agent-hooks control engine.

This module makes the framework-neutral
`agent-hooks <https://github.com/responsibleai/agent-hooks>`_ contract crewAI's
control engine. You register ``Interceptor`` objects (policy engines, content
filters, rate limiters, egress guards) once; crewAI's ``dispatch`` then
delegates its lifecycle points to the agent-hooks emitter, which runs them as
the authoritative control layer after any crewAI-native hooks and records every
decision as an auditable ``InterceptionRecord``.

Design notes:
    - ``agent-hooks`` is an **optional** dependency. It ships a compiled native
      core and is not required by crewAI; this module imports it lazily and
      raises an actionable :class:`ImportError` (see :data:`HAS_AGENT_HOOKS`)
      only when the engine is actually activated.
    - The ``agent-hooks`` emitter is asynchronous. crewAI's hook seams are
      synchronous and may be called from either sync or async code, so the
      engine drives the emitter on a single dedicated background event loop
      (see :class:`_EmitterLoop`). Serializing every emission on one loop
      preserves the emitter's per-session ``sequence``/record atomicity
      guarantees (agent-hooks spec Section 12.2).
    - Verdicts are mapped to crewAI's hook semantics and **fail closed**: a
      ``deny`` blocks the guarded action, a ``transform`` rewrites the
      interceptable value, and any engine-internal error blocks rather than
      silently allowing.

Failure modes:
    - Interceptor errors/timeouts are turned into fail-closed denies by the
      emitter itself (agent-hooks spec Section 6.3); the engine surfaces them as
      a blocked call.
    - An emission that raises unexpectedly is logged and treated as a deny.

Interception-point mapping (crewAI -> agent-hooks):
    ``PRE_TOOL_CALL`` -> ``pre_tool_call``,
    ``POST_TOOL_CALL`` -> ``post_tool_call``,
    ``PRE_MODEL_CALL`` -> ``pre_model_call``,
    ``POST_MODEL_CALL`` -> ``post_model_call``,
    ``INPUT`` -> ``input``,
    ``OUTPUT`` -> ``output``,
    ``EXECUTION_START`` -> ``agent_startup``,
    ``EXECUTION_END`` -> ``agent_shutdown``.

crewAI's ``PRE_STEP``/``POST_STEP`` points have no agent-hooks equivalent and
remain crewAI-native (they are not governed by the engine).
"""

# The agent-hooks ``AgentContext`` is dynamic wire JSON (``dict[str, Any]``);
# reading the post-transform ``target`` yields values a strict checker cannot
# infer past ``Unknown``. Each is runtime-validated (``isinstance``) before use,
# so relax only the unknown-type diagnostics for this boundary module. mypy
# ``--strict`` (the repository's type gate) still applies in full.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import functools
import hashlib
import json
import logging
import math
import threading
from typing import TYPE_CHECKING, Any, Final, TypeVar

from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from crewai.hooks.contexts import (
        ExecutionEndContext,
        ExecutionStartContext,
        InputContext,
        OutputContext,
    )
    from crewai.hooks.llm_hooks import LLMCallHookContext
    from crewai.hooks.tool_hooks import ToolCallHookContext


# The optional agent-hooks package is imported lazily at runtime; its types are
# surfaced as ``Any`` in this module's signatures so the module type-checks
# under crewAI's strict mypy, which runs without the optional dependency
# installed. The real objects are imported inside the methods that use them.
AgentContext = Any
AgentContextBuilder = Any
ApprovalResolver = Any
CompositionConfig = Any
EnforcementMode = Any
IdentityProvider = Any
InterceptionEmitter = Any
InterceptionRecord = Any
Interceptor = Any


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _probe_agent_hooks() -> tuple[bool, Exception | None]:
    """Detect agent-hooks availability, including native-core load failures.

    Returns:
        ``(available, error)`` where ``error`` is the captured import failure
        when the package (or its compiled core) could not be loaded.
    """
    try:
        import agent_hooks  # type: ignore[import-not-found]  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except Exception as exc:  # ImportError, or a native-core load failure
        return False, exc
    return True, None


# ``agent-hooks`` is an optional dependency: probe it once at import time so
# callers can branch on availability without paying a repeated import cost, and
# so importing this module never fails when it is absent.
_availability = _probe_agent_hooks()
#: Whether the optional ``agent_hooks`` dependency is importable.
HAS_AGENT_HOOKS: Final[bool] = _availability[0]
_IMPORT_ERROR: Final[Exception | None] = _availability[1]


#: RECOMMENDED per-interceptor timeout in seconds (agent-hooks spec Section 7).
DEFAULT_TIMEOUT: Final[float] = 5.0

#: Framework identifier stamped on every emitted ``AgentContext``.
_FRAMEWORK: Final[str] = "crewai"

#: Default identity provider name understood by the emitter (agent-hooks Section 10.2).
_DEFAULT_IDENTITY: Final[str] = "jcs-sha256"

#: ``source`` attached to a :class:`HookAborted` raised by the engine.
_SOURCE: Final[str] = "agent-hooks"

#: Reserved reason used when the engine itself fails and must fail closed.
_ENGINE_FAILED: Final[str] = "host_error:engine_failed"

#: The eight agent-hooks lifecycle points the engine governs, in order.
DEFAULT_POINTS: Final[tuple[InterceptionPoint, ...]] = (
    InterceptionPoint.EXECUTION_START,
    InterceptionPoint.INPUT,
    InterceptionPoint.PRE_MODEL_CALL,
    InterceptionPoint.POST_MODEL_CALL,
    InterceptionPoint.PRE_TOOL_CALL,
    InterceptionPoint.POST_TOOL_CALL,
    InterceptionPoint.OUTPUT,
    InterceptionPoint.EXECUTION_END,
)

#: Post points whose fail-closed action is a blocked *result* string rather than
#: a raised :class:`HookAborted` (crewAI post seams cannot raise).
_POST_POINTS: Final[frozenset[InterceptionPoint]] = frozenset(
    {InterceptionPoint.POST_TOOL_CALL, InterceptionPoint.POST_MODEL_CALL}
)

_INSTALL_HINT: Final[str] = (
    "agent-hooks is not installed (or its native core failed to load). It is an "
    "optional dependency with a compiled core; install it from source:\n"
    '    pip install "agent-hooks-sdk @ '
    'git+https://github.com/responsibleai/agent-hooks.git#subdirectory=sdk/python"\n'
    "See https://github.com/responsibleai/agent-hooks for details."
)


def _require() -> None:
    """Raise an actionable :class:`ImportError` when agent-hooks is unavailable.

    Raises:
        ImportError: If the ``agent_hooks`` package (or its native core) could
            not be imported at module load time.
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(_INSTALL_HINT) from _IMPORT_ERROR


def _json_safe(value: object, _seen: frozenset[int] = frozenset()) -> Any:
    """Coerce ``value`` into a JSON-serializable form for an ``AgentContext``.

    Model output and tool results are untrusted and may contain objects the
    agent-hooks wire format cannot carry (which would fail the emission closed).
    This normalizes them at the boundary so an interceptor always sees a stable,
    inspectable value. Reference cycles are broken with a ``"<cycle>"``
    placeholder so a self-referential container cannot raise ``RecursionError``
    (which, raised before :meth:`AgentHooksEngine._decide`, would fail open).

    Args:
        value: Any Python value taken from a crewAI hook context.
        _seen: Internal set of ``id()``s of containers currently being encoded,
            used to detect cycles. Callers should not pass it.

    Returns:
        A value composed only of ``dict``/``list``/``str``/``int``/``float``/
        ``bool``/``None`` (non-finite floats and unknown objects become ``str``).
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        if id(value) in _seen:
            return "<cycle>"
        seen = _seen | {id(value)}
        return {str(k): _json_safe(v, seen) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        if id(value) in _seen:
            return "<cycle>"
        seen = _seen | {id(value)}
        return [_json_safe(v, seen) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_safe(dump(mode="json"), _seen)
        except Exception:
            logger.debug("model_dump() failed while normalizing %s", type(value))
    return str(value)


def _to_text(value: Any) -> str:
    """Render a (possibly transformed) interceptable value as a result string."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _transformed_payload(target: Any) -> Any:
    """Extract the replacement payload from a transformed input/output target.

    The ``input``/``output`` contexts wrap the value in a ``{"content": ...}``
    envelope; a transform may rewrite the envelope or the content directly.
    """
    if isinstance(target, dict):
        if "content" in target:
            return target["content"]
        return target
    return target


def _blocked_result(reason: str) -> str:
    """Fail-closed replacement returned when a *post* point denies a result."""
    return f"[blocked by agent-hooks: {reason}]"


def _compose_reason(reason: str | None, message: str | None) -> str:
    """Combine an interceptor verdict's ``reason`` and ``message``.

    A verdict may carry a short machine ``reason`` (e.g. a policy reason code)
    and a human ``message``. Both are surfaced so callers — and, ultimately,
    crewAI customers who receive the blocked-call error — see the full policy
    decision rather than just an opaque code. Falls back gracefully when only
    one (or neither) is present.
    """
    reason_text = (reason or "").strip()
    message_text = (message or "").strip()
    if reason_text and message_text and message_text != reason_text:
        return f"{reason_text}: {message_text}"
    return reason_text or message_text or "blocked by agent-hooks interceptor"


def _agent_ids(agent: Any) -> tuple[str, str | None]:
    """Derive ``(agent_id, agent_name)`` from a crewAI agent (or ``None``)."""
    if agent is None:
        return "crewai-agent", None
    ident = getattr(agent, "id", None)
    role = getattr(agent, "role", None)
    role_str = str(role) if role else None
    agent_id = str(ident) if ident is not None else (role_str or "crewai-agent")
    return agent_id, role_str


def _session_id(*, crew: Any = None, flow: Any = None, agent: Any = None) -> str:
    """Derive a stable session id, preferring crew/flow identity over agent."""
    for obj in (crew, flow):
        ident = getattr(obj, "id", None)
        if ident is not None:
            return str(ident)
    ident = getattr(agent, "id", None)
    if ident is not None:
        return f"agent:{ident}"
    return "crewai-session"


def _model_id(llm: Any) -> str:
    """Best-effort model identifier from a crewAI LLM reference."""
    for attr in ("model", "model_name", "id"):
        val = getattr(llm, attr, None)
        if isinstance(val, str) and val:
            return val
    return "unknown"


def _registered_tools(crew: Any, agent: Any = None) -> list[str]:
    """Collect declared tool names for the ``agent_startup`` context."""
    names: list[str] = []
    for source in (agent, crew):
        tools = getattr(source, "tools", None) or []
        for tool in tools:
            name = getattr(tool, "name", None)
            if name:
                names.append(str(name))
    return names


def _agent_envelope(agent: Any, framework: str) -> dict[str, Any]:
    """Build the agent-hooks ``agent`` envelope for the agent that acted.

    The per-session builder bakes in one agent id, but a crew/flow session
    drives many agents; the engine stamps this envelope onto each built context
    so an interceptor and the audit record attribute the emission to the agent
    that actually acted, not the session's first agent.
    """
    agent_id, agent_name = _agent_ids(agent)
    envelope: dict[str, Any] = {"id": agent_id, "framework": framework}
    if agent_name:
        envelope["name"] = agent_name
    return envelope


def _tool_call_id(session_id: str, name: str, args: Any) -> str:
    """Derive a stable per-invocation tool-call id.

    crewAI builds separate pre/post hook contexts for one tool invocation, so a
    random id cannot correlate them. A digest over the session, tool name, and
    (JSON-safe) arguments yields the same id at both seams, letting audit sinks
    and post-call policies pair the pre and post records for one call.
    """
    payload = json.dumps(
        {"session": session_id, "name": name, "args": args},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _model_response_parts(response: Any) -> tuple[Any, list[Any], str]:
    """Split a model response into ``(content, tool_calls, finish_reason)``.

    A model-boundary tool-use policy must see the tool calls the model
    requested. When the response carries them (a structured mapping) they are
    surfaced to the interceptor; a plain-string response yields no tool calls.
    """
    if isinstance(response, dict):
        raw_calls = response.get("tool_calls")
        tool_calls = (
            [_json_safe(call) for call in raw_calls]
            if isinstance(raw_calls, list)
            else []
        )
        finish = response.get("finish_reason")
        finish_reason = (
            finish
            if isinstance(finish, str)
            else ("tool_calls" if tool_calls else "stop")
        )
        return _json_safe(response.get("content", "")), tool_calls, finish_reason
    return _json_safe(response), [], "stop"


class _EmitterLoop:
    """Drive the async agent-hooks emitter from crewAI's sync hook seams.

    Owns one daemon thread running a private event loop. All emissions are
    submitted to that loop and awaited synchronously, which both bridges the
    sync/async boundary and serializes emissions so the emitter's per-session
    ``sequence`` and record buffer stay consistent.
    """

    __slots__ = ("_loop", "_thread")

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="agent-hooks-emitter", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, _T], timeout: float | None) -> _T:
        """Submit ``coro`` to the background loop and block for its result.

        Args:
            coro: The coroutine to execute (an ``emit_unchecked`` call).
            timeout: Optional wall-clock bound in seconds; ``None`` waits
                indefinitely (the emitter enforces its own interceptor timeout).

        Returns:
            The coroutine's result.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    def close(self) -> None:
        """Stop the background loop, releasing any in-flight emission.

        Pending emissions are cancelled first so a thread blocked in
        :meth:`run` on ``future.result(None)`` is released (with a
        ``CancelledError``) instead of deadlocking; only then is the loop
        stopped and its thread joined.
        """
        if self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._cancel_pending(), self._loop
                ).result(timeout=5.0)
            except Exception:
                logger.debug("agent-hooks loop drain did not complete cleanly")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._loop.is_running():
            self._loop.close()

    @staticmethod
    async def _cancel_pending() -> None:
        """Cancel every other task on the loop and await their unwinding."""
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@dataclass(frozen=True, slots=True)
class _Decision:
    """The engine's normalized view of one emission outcome."""

    proceeds: bool
    is_transform: bool
    reason: str


class AgentHooksEngine:
    """crewAI's agent-hooks control engine.

    Wraps an agent-hooks ``InterceptionEmitter`` and exposes a per-point adapter
    that :mod:`crewai.hooks.dispatch` runs as the authoritative final hook for
    the eight lifecycle points. Construct with one or more ``Interceptor``
    objects (or use :func:`use_agent_hooks`), then :meth:`activate` to install it
    as the process governor. Every emission is recorded; inspect :attr:`records`
    or supply a ``record_sink`` for audit.

    Safe to use as a context manager; exiting calls :meth:`close`, which
    deactivates the engine and stops the background loop.
    """

    def __init__(
        self,
        *interceptors: Interceptor,
        mode: EnforcementMode | None = None,
        composition: CompositionConfig | None = None,
        resolver: ApprovalResolver | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
        identity_provider: str | IdentityProvider | None = _DEFAULT_IDENTITY,
        record_sink: Callable[[InterceptionRecord], None] | None = None,
        points: Sequence[InterceptionPoint] | None = None,
        framework: str = _FRAMEWORK,
        emit_timeout: float | None = None,
    ) -> None:
        """Build the underlying emitter and register the interceptors.

        Args:
            interceptors: agent-hooks interceptors to enforce, in order.
            mode: ``ENFORCE`` (default) or ``EVALUATE_ONLY`` (record without
                blocking).
            composition: Composition profile/knobs; defaults to
                ``sequential/first_deny``.
            resolver: Optional approval resolver for liftable denies (Section 9).
            timeout: Per-interceptor timeout in seconds (``None`` disables it).
            identity_provider: Identity provider name, custom provider, or
                ``None`` for identity-unbound records.
            record_sink: Optional callback invoked with every record.
            points: Lifecycle points to govern; defaults to :data:`DEFAULT_POINTS`.
            framework: Framework id stamped on every context.
            emit_timeout: Optional wall-clock bound per emission; ``None``
                relies on the emitter's own interceptor timeout.

        Raises:
            ImportError: If agent-hooks is not installed.
        """
        _require()
        from agent_hooks import (
            EnforcementMode as _EnforcementMode,
            InterceptionEmitter as _InterceptionEmitter,
        )

        emitter: Any = _InterceptionEmitter(
            mode=mode if mode is not None else _EnforcementMode.ENFORCE,
            resolver=resolver,
            timeout=timeout,
            composition=composition,
            identity_provider=identity_provider,
        )
        for interceptor in interceptors:
            emitter.register(interceptor)
        if record_sink is not None:
            emitter.set_record_sink(record_sink)

        self._emitter: InterceptionEmitter = emitter
        self._framework = framework
        self._emit_timeout = emit_timeout
        self._loop = _EmitterLoop()
        self._builders: dict[str, AgentContextBuilder] = {}
        self._builders_lock = threading.Lock()
        self._points: frozenset[InterceptionPoint] = frozenset(
            points if points is not None else DEFAULT_POINTS
        )
        self._active = False
        self._closed = False
        raw_adapters: dict[InterceptionPoint, Callable[[Any], Any]] = {
            InterceptionPoint.PRE_TOOL_CALL: self._pre_tool_call,
            InterceptionPoint.POST_TOOL_CALL: self._post_tool_call,
            InterceptionPoint.PRE_MODEL_CALL: self._pre_model_call,
            InterceptionPoint.POST_MODEL_CALL: self._post_model_call,
            InterceptionPoint.INPUT: self._input,
            InterceptionPoint.OUTPUT: self._output,
            InterceptionPoint.EXECUTION_START: self._execution_start,
            InterceptionPoint.EXECUTION_END: self._execution_end,
        }
        # Every adapter builds its context outside _decide()'s fail-closed
        # guard, so wrap each so an unexpected build error fails closed instead
        # of escaping to dispatch._invoke_hook (which would swallow it).
        self._adapters: dict[InterceptionPoint, Callable[[Any], Any]] = {
            point: self._wrap_fail_closed(point, adapter)
            for point, adapter in raw_adapters.items()
        }

    # -- lifecycle ------------------------------------------------------------

    @property
    def emitter(self) -> InterceptionEmitter:
        """The underlying agent-hooks emitter (for advanced configuration)."""
        return self._emitter

    @property
    def records(self) -> list[InterceptionRecord]:
        """All interception records emitted so far, in order."""
        return list(self._emitter.results)

    def take_records(self) -> list[InterceptionRecord]:
        """Drain and return the buffered interception records."""
        return list(self._emitter.take_records())

    def adapter_for(self, point: InterceptionPoint) -> Callable[[Any], Any] | None:
        """Return the governing adapter for ``point``, or ``None`` if not governed.

        Called by :func:`crewai.hooks.dispatch.dispatch` (via the installed
        governor) to append the engine as the authoritative final hook.
        """
        if point not in self._points:
            return None
        return self._adapters.get(point)

    def activate(self) -> AgentHooksEngine:
        """Install this engine as crewAI's control governor (idempotent)."""
        if not self._active:
            from crewai.hooks.dispatch import set_governor

            set_governor(self.adapter_for)
            self._active = True
            logger.debug(
                "agent-hooks engine activated on %d point(s)", len(self._points)
            )
        return self

    def deactivate(self) -> None:
        """Remove this engine as the governor if it is currently installed."""
        if self._active:
            from crewai.hooks.dispatch import clear_governor, get_governor

            if get_governor() == self.adapter_for:
                clear_governor()
            self._active = False

    def close(self) -> None:
        """Deactivate the engine and stop the background emitter loop (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.deactivate()
        self._loop.close()

    def __enter__(self) -> AgentHooksEngine:
        return self.activate()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- emission core --------------------------------------------------------

    def _builder(
        self, *, crew: Any = None, flow: Any = None, agent: Any = None
    ) -> AgentContextBuilder:
        """Return the per-session context builder, creating it on first use."""
        session_id = _session_id(crew=crew, flow=flow, agent=agent)
        with self._builders_lock:
            builder = self._builders.get(session_id)
            if builder is None:
                from agent_hooks import AgentContextBuilder

                agent_id, agent_name = _agent_ids(agent)
                builder = AgentContextBuilder(
                    agent_id=agent_id,
                    framework=self._framework,
                    session_id=session_id,
                    agent_name=agent_name,
                )
                self._builders[session_id] = builder
            return builder

    def _decide(self, ctx: AgentContext) -> _Decision:
        """Run one emission and normalize the outcome, failing closed on error.

        Args:
            ctx: The fully built agent-hooks context. Mutated in place by the
                emitter when a transform is applied (enforce mode).

        Returns:
            A :class:`_Decision` describing whether the action proceeds and
            whether a transform was applied.
        """
        try:
            record = self._loop.run(
                self._emitter.emit_unchecked(ctx), self._emit_timeout
            )
        except Exception:
            logger.exception(
                "agent-hooks emission failed at %s; failing closed",
                ctx.get("interception_point"),
            )
            return _Decision(proceeds=False, is_transform=False, reason=_ENGINE_FAILED)
        verdict = record.verdict
        reason = _compose_reason(verdict.reason, verdict.message)
        return _Decision(
            proceeds=record.proceeds,
            is_transform=record.proceeds and verdict.decision.value == "transform",
            reason=reason,
        )

    # -- per-point adapters ---------------------------------------------------

    def _wrap_fail_closed(
        self, point: InterceptionPoint, adapter: Callable[[Any], Any]
    ) -> Callable[[Any], Any]:
        """Wrap an adapter so any unexpected error fails closed.

        Each adapter builds its context (``_json_safe`` / ``builder.*``) outside
        :meth:`_decide`'s guard. Without this wrapper an error there (e.g. a
        ``RecursionError`` from a cyclic tool input) would escape to
        :func:`crewai.hooks.dispatch._invoke_hook`, which swallows non-abort
        exceptions and lets the guarded action proceed **ungoverned**. A
        legitimate :class:`HookAborted` still propagates unchanged; a *post*
        point fails the result closed with a blocked marker (its seam cannot
        raise), every other point raises a fail-closed abort.
        """
        is_post = point in _POST_POINTS

        @functools.wraps(adapter)
        def guarded(ctx: Any) -> Any:
            try:
                return adapter(ctx)
            except HookAborted:
                raise
            except Exception:
                logger.exception(
                    "agent-hooks adapter for %s failed; failing closed", point.value
                )
                if is_post:
                    return _blocked_result(_ENGINE_FAILED)
                raise HookAborted(reason=_ENGINE_FAILED, source=_SOURCE) from None

        return guarded

    def _pre_tool_call(self, ctx: ToolCallHookContext) -> None:
        """Adapt ``PRE_TOOL_CALL``: deny blocks the call; transform rewrites args."""
        flow = getattr(ctx, "flow", None)
        args = _json_safe(dict(ctx.tool_input))
        session_id = _session_id(crew=ctx.crew, flow=flow, agent=ctx.agent)
        builder = self._builder(crew=ctx.crew, flow=flow, agent=ctx.agent)
        agent_ctx = builder.pre_tool_call(
            call_id=_tool_call_id(session_id, ctx.tool_name, args),
            name=ctx.tool_name,
            args=args,
        )
        agent_ctx["agent"] = _agent_envelope(ctx.agent, self._framework)
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            raise HookAborted(reason=decision.reason, source=_SOURCE)
        if decision.is_transform:
            new_args = agent_ctx.get("target")
            if isinstance(new_args, dict):
                ctx.tool_input.clear()
                ctx.tool_input.update(new_args)

    def _post_tool_call(self, ctx: ToolCallHookContext) -> str | None:
        """Adapt ``POST_TOOL_CALL``: deny fails the result closed; transform rewrites it."""
        flow = getattr(ctx, "flow", None)
        args = _json_safe(dict(ctx.tool_input))
        session_id = _session_id(crew=ctx.crew, flow=flow, agent=ctx.agent)
        builder = self._builder(crew=ctx.crew, flow=flow, agent=ctx.agent)
        agent_ctx = builder.post_tool_call(
            call_id=_tool_call_id(session_id, ctx.tool_name, args),
            name=ctx.tool_name,
            args=args,
            value=_json_safe(ctx.tool_result),
            is_error=False,
        )
        agent_ctx["agent"] = _agent_envelope(ctx.agent, self._framework)
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            return _blocked_result(decision.reason)
        if decision.is_transform:
            return _to_text(agent_ctx.get("target"))
        return None

    def _pre_model_call(self, ctx: LLMCallHookContext) -> None:
        """Adapt ``PRE_MODEL_CALL``: deny blocks the call; transform rewrites messages."""
        flow = getattr(ctx, "flow", None)
        builder = self._builder(crew=ctx.crew, flow=flow, agent=ctx.agent)
        agent_ctx = builder.pre_model_call(
            model_id=_model_id(getattr(ctx, "llm", None)),
            messages=_json_safe(list(ctx.messages)),
        )
        agent_ctx["agent"] = _agent_envelope(ctx.agent, self._framework)
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            raise HookAborted(reason=decision.reason, source=_SOURCE)
        if decision.is_transform:
            new_messages = agent_ctx.get("target")
            if isinstance(new_messages, list):
                ctx.messages.clear()
                ctx.messages.extend(new_messages)

    def _post_model_call(self, ctx: LLMCallHookContext) -> str | None:
        """Adapt ``POST_MODEL_CALL``: deny fails the response closed; transform rewrites it."""
        flow = getattr(ctx, "flow", None)
        content, tool_calls, finish_reason = _model_response_parts(ctx.response)
        builder = self._builder(crew=ctx.crew, flow=flow, agent=ctx.agent)
        agent_ctx = builder.post_model_call(
            model_id=_model_id(getattr(ctx, "llm", None)),
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        agent_ctx["agent"] = _agent_envelope(ctx.agent, self._framework)
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            return _blocked_result(decision.reason)
        if decision.is_transform:
            target = agent_ctx.get("target")
            if isinstance(target, dict):
                return _to_text(target.get("content"))
            return _to_text(target)
        return None

    def _input(self, ctx: InputContext) -> Any:
        """Adapt ``INPUT``: deny blocks execution; transform replaces the inputs."""
        builder = self._builder(crew=ctx.crew, flow=ctx.flow)
        agent_ctx = builder.input(content=_json_safe(ctx.payload))
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            raise HookAborted(reason=decision.reason, source=_SOURCE)
        if decision.is_transform:
            return _transformed_payload(agent_ctx.get("target"))
        return None

    def _output(self, ctx: OutputContext) -> Any:
        """Adapt ``OUTPUT``: deny blocks the final output; transform replaces plain payloads.

        A transform is only applied when the crewAI payload is a plain ``str``
        or ``dict``; rich ``CrewOutput`` objects cannot be safely reconstructed
        from the transformed value, so only ``deny`` acts on them.
        """
        builder = self._builder(crew=ctx.crew, flow=ctx.flow)
        payload = ctx.payload
        content = getattr(payload, "raw", None)
        agent_ctx = builder.output(
            content=_json_safe(content if content is not None else payload)
        )
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            raise HookAborted(reason=decision.reason, source=_SOURCE)
        if decision.is_transform and isinstance(payload, (str, dict)):
            return _transformed_payload(agent_ctx.get("target"))
        return None

    def _execution_start(self, ctx: ExecutionStartContext) -> None:
        """Adapt ``EXECUTION_START`` -> ``agent_startup``: deny aborts the run."""
        builder = self._builder(crew=ctx.crew, flow=ctx.flow)
        agent_ctx = builder.agent_startup(
            tools_registered=_registered_tools(ctx.crew, ctx.agent),
            inputs=_json_safe(ctx.payload),
        )
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            raise HookAborted(reason=decision.reason, source=_SOURCE)

    def _execution_end(self, ctx: ExecutionEndContext) -> None:
        """Adapt ``EXECUTION_END`` -> ``agent_shutdown``: records the run's end."""
        builder = self._builder(crew=ctx.crew, flow=ctx.flow)
        agent_ctx = builder.agent_shutdown(reason=ctx.status)
        decision = self._decide(agent_ctx)
        if not decision.proceeds:
            raise HookAborted(reason=decision.reason, source=_SOURCE)


_active_engine: AgentHooksEngine | None = None


def use_agent_hooks(
    *interceptors: Interceptor,
    mode: EnforcementMode | None = None,
    composition: CompositionConfig | None = None,
    resolver: ApprovalResolver | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    identity_provider: str | IdentityProvider | None = _DEFAULT_IDENTITY,
    record_sink: Callable[[InterceptionRecord], None] | None = None,
    points: Sequence[InterceptionPoint] | None = None,
    framework: str = _FRAMEWORK,
    emit_timeout: float | None = None,
) -> AgentHooksEngine:
    """Make agent-hooks crewAI's active control engine and return it.

    This is the one-call entry point: crewAI's ``dispatch`` delegates the eight
    lifecycle points to the agent-hooks emitter, running the given interceptors
    as the authoritative control layer after any crewAI-native hooks::

        from crewai.hooks import use_agent_hooks

        engine = use_agent_hooks(MyPolicy(), ContentFilter())
        try:
            crew.kickoff(inputs=...)
        finally:
            engine.close()

    A previously active engine is closed first. Also usable as a context manager
    (``with use_agent_hooks(...) as engine:``).

    Args:
        interceptors: agent-hooks interceptors to enforce, in order.
        mode: ``ENFORCE`` (default) or ``EVALUATE_ONLY``.
        composition: Composition profile/knobs (default ``sequential/first_deny``).
        resolver: Optional approval resolver for liftable denies.
        timeout: Per-interceptor timeout in seconds (``None`` disables it).
        identity_provider: Identity provider name, custom provider, or ``None``.
        record_sink: Optional callback invoked with every record.
        points: Lifecycle points to govern (default :data:`DEFAULT_POINTS`).
        framework: Framework id stamped on every context.
        emit_timeout: Optional wall-clock bound per emission.

    Returns:
        The active :class:`AgentHooksEngine`. Call :meth:`AgentHooksEngine.close`
        (or use it as a context manager) to deactivate and release resources.

    Raises:
        ImportError: If agent-hooks is not installed.
    """
    global _active_engine
    if _active_engine is not None:
        _active_engine.close()
    engine = AgentHooksEngine(
        *interceptors,
        mode=mode,
        composition=composition,
        resolver=resolver,
        timeout=timeout,
        identity_provider=identity_provider,
        record_sink=record_sink,
        points=points,
        framework=framework,
        emit_timeout=emit_timeout,
    )
    engine.activate()
    _active_engine = engine
    return engine


def disable_agent_hooks() -> None:
    """Deactivate and close the active agent-hooks control engine, if any."""
    global _active_engine
    if _active_engine is not None:
        _active_engine.close()
        _active_engine = None


def active_engine() -> AgentHooksEngine | None:
    """Return the currently active :class:`AgentHooksEngine`, or ``None``."""
    return _active_engine


__all__ = [
    "DEFAULT_POINTS",
    "DEFAULT_TIMEOUT",
    "HAS_AGENT_HOOKS",
    "AgentHooksEngine",
    "active_engine",
    "disable_agent_hooks",
    "use_agent_hooks",
]
