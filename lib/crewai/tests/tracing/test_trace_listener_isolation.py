"""`TraceCollectionListener` is a process-wide singleton, so its state leaks.

`TraceBatchManager` is cached on the listener class and `_initialized`
short-circuits `__init__`, so anything a test leaves on the batch manager applies
to every later test in the same xdist worker. When `trace_batch_id` leaks, trace
POSTs move from `/tracing/ephemeral/batches` to
`/tracing/batches/<leaked-id>/events`, the recorded cassette stops matching, and
the failure lands in whichever unrelated test happens to run later — the symptom
names neither the leak nor the test that caused it.

The autouse `reset_tracing_state` fixture in the root `conftest.py` drops the
singleton and the tracing context vars after every test. These are canaries for
it: run on their own they pass trivially, but under the full suite in random
order they fail if the fixture stops working.

The context vars matter as much as the singleton. Dropping the listener alone
makes its replacement re-register `on_task_failed` while `_tracing_enabled` is
still set, which breaks `test_task_failure_instrumentation` (one handler per
event). Clearing the context is what makes replacing the listener safe.
"""

from __future__ import annotations

from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener


def test_batch_manager_starts_clean() -> None:
    """No earlier test may leave batch state on the shared singleton.

    Regression: `test_nested_agent_executor_flow_does_not_finalize_parent_batch`
    set `trace_batch_id = "debug-trace-batch"` and never restored it, which broke
    `test_trace_calls_when_enabled_via_env` several hundred tests later.
    """
    manager = TraceCollectionListener().batch_manager

    assert manager.trace_batch_id is None
    assert manager.current_batch is None
    assert manager.batch_owner_type is None
    assert manager.batch_owner_id is None
    assert manager.event_buffer == []
    assert manager.defer_session_finalization is False


def test_listener_can_register_handlers_again() -> None:
    """A leaked listener must not block later handler registration.

    `cleanup_event_handlers` wipes the bus between tests, but `setup_listeners`
    returns early when `_listeners_setup` is already set (`trace_listener.py:208`).
    A listener surviving a tracing-enabled test would therefore register nothing
    for the rest of the worker — tracing silently collects no events instead of
    failing visibly.
    """
    assert TraceCollectionListener()._listeners_setup is False
