"""Per-run execution identity for OSS traces.

Wharf keys spans by ``crewai.execution_uuid``. Enterprise stamps that from the
Celery ``kickoff_id`` inside ``telemetry_session``. Standalone OSS has no such
session, so crew/flow ``kickoff`` creates a uuid for the **outermost** run and
nested kickoffs (crew-in-flow, AgentExecutor, child flows) inherit it via
contextvars on the execution thread.

Minting lives on the kickoff call path (not the event bus): bus handlers run
on worker threads and cannot publish contextvars back to the user thread.

Enterprise (or any host) can call :func:`set_execution_uuid` before kickoff;
:func:`begin_execution` will not overwrite it.
"""

from __future__ import annotations

from contextlib import ExitStack
import contextvars
from dataclasses import dataclass
import os
import sys
from types import TracebackType
from typing import TYPE_CHECKING
from uuid import uuid4


if TYPE_CHECKING:
    from crewai.telemetry.tracing.session import TraceSession


@dataclass
class ExecutionTrace:
    """Trace lifetime that can be rebound between deferred conversational turns."""

    session: TraceSession
    cleanup: ExitStack
    closed: bool = False

    def finish(
        self,
        error_type: type[BaseException] | None = None,
        error: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if self.closed:
            return
        self.closed = True
        with self.session.activate():
            self.cleanup.__exit__(error_type, error, traceback)


_current_execution_uuid: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "crewai_execution_uuid", default=None
)
_execution_tracing: contextvars.ContextVar[
    tuple[ExecutionTrace, ExitStack, BaseException | None, TracebackType | None] | None
] = contextvars.ContextVar("crewai_execution_tracing", default=None)


def get_execution_uuid() -> str | None:
    """Return the active execution uuid, if any."""
    return _current_execution_uuid.get()


def set_execution_uuid(execution_uuid: str) -> contextvars.Token[str | None]:
    """Bind an execution uuid for the current context.

    Use this from enterprise / hosts that already own the run id (e.g. Celery
    ``kickoff_id``). Overwrites any previously bound value.
    """
    if not execution_uuid:
        raise ValueError("execution_uuid must be a non-empty string")
    return _current_execution_uuid.set(execution_uuid)


def clear_execution_uuid(token: contextvars.Token[str | None]) -> None:
    """Restore the execution uuid that was active before ``token`` was issued.

    ``token`` is required so this cannot wipe an outer kickoff's uuid.
    """
    _current_execution_uuid.reset(token)


def begin_execution(
    execution_uuid: str | None = None,
    *,
    tracing: bool | None = None,
    trace_session: ExecutionTrace | None = None,
) -> contextvars.Token[str | None] | None:
    """Start an execution context unless one is already active.

    The outermost crew or flow receives a new uuid by default. Nested
    executions inherit the active value and return no reset token.
    """
    if _current_execution_uuid.get() is not None:
        return None
    if trace_session is not None and not trace_session.closed:
        execution_uuid = trace_session.session.context.kickoff_id
    else:
        trace_session = None
        execution_uuid = execution_uuid or str(uuid4())
    token = set_execution_uuid(execution_uuid)
    try:
        if trace_session is None:
            _start_tracing(execution_uuid, tracing)
        else:
            _activate_tracing(trace_session)
    except BaseException:
        clear_execution_uuid(token)
        raise
    return token


def _start_tracing(execution_uuid: str, tracing: bool | None) -> None:
    from crewai.events.listeners.tracing.utils import should_enable_tracing
    from crewai.telemetry.tracing.context import get_trace_session

    if (
        get_trace_session() is not None
        or os.getenv("OTEL_SDK_DISABLED", "").lower() == "true"
    ):
        return
    if not should_enable_tracing(override=tracing):
        return
    from crewai.telemetry.tracing.grants import (
        GrantSpanExporter,
        TraceGrantClient,
        tracing_credential,
    )
    from crewai.telemetry.tracing.session import TraceSession

    stack = ExitStack()
    amp_credential = tracing_credential()
    if amp_credential is None:
        from crewai.telemetry.tracing.ephemeral import ephemeral_tracing

        session = stack.enter_context(ephemeral_tracing(execution_uuid))
    else:
        client = TraceGrantClient(amp_credential)
        grant = client.create(execution_uuid)
        session = TraceSession(grant.execution_uuid, [GrantSpanExporter(client, grant)])
        stack.callback(session.shutdown)
    _activate_tracing(ExecutionTrace(session, stack))


def _activate_tracing(tracing: ExecutionTrace) -> None:
    activation = ExitStack()
    activation.enter_context(tracing.session.activate())
    # Kickoff may itself be called inside an except block (including the sync
    # Flow wrapper's event-loop detection). That is not a failure of this run.
    _, ambient_error, ambient_traceback = sys.exc_info()
    _execution_tracing.set((tracing, activation, ambient_error, ambient_traceback))


def end_execution(
    token: contextvars.Token[str | None] | None, *, defer: bool = False
) -> ExecutionTrace | None:
    """End an execution context owned by the current kickoff.

    Nested kickoffs pass ``None`` and leave the outer uuid in place.
    """
    if token is not None:
        tracing = _execution_tracing.get()
        try:
            if tracing is not None:
                lifetime, activation, ambient_error, ambient_traceback = tracing
                error_type, error, traceback = sys.exc_info()
                if error is ambient_error and traceback is ambient_traceback:
                    error_type, error, traceback = None, None, None
                try:
                    if defer and error is None and not lifetime.closed:
                        return lifetime
                    lifetime.finish(error_type, error, traceback)
                finally:
                    activation.close()
        finally:
            _execution_tracing.set(None)
            clear_execution_uuid(token)
    return None
