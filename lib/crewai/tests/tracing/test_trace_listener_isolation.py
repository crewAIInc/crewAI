"""`TraceCollectionListener` is a process-wide singleton, so its state leaks.

`TraceBatchManager` is cached on the listener class and `_initialized`
short-circuits `__init__`, so anything a test leaves on the batch manager applies
to every later test in the same xdist worker. When `trace_batch_id` leaks, trace
POSTs move from `/tracing/ephemeral/batches` to
`/tracing/batches/<leaked-id>/events`, the recorded cassette stops matching, and
the failure lands in whichever unrelated test happens to run later — the symptom
names neither the leak nor the test that caused it.

The autouse `reset_trace_batch_state` fixture in the root `conftest.py` clears
that state after every test. This is a canary for it: run on its own it passes
trivially, but under the full suite in random order it fails if the fixture stops
working.

The fixture clears the manager in place rather than dropping the singleton. A
fresh listener re-runs `setup_listeners` and re-registers its handlers, which
breaks `test_task_failure_instrumentation` — that test requires exactly one
handler per event, and the re-registered `on_task_failed` makes two. Handler
cleanup is already `cleanup_event_handlers`' job; this fixture owns batch state
only.
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
