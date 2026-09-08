"""`TraceCollectionListener` is a process-wide singleton, so its state leaks.

`TraceBatchManager` is cached on the listener class and `_initialized`
short-circuits `__init__`, so anything a test leaves on the batch manager applies
to every later test in the same xdist worker. When `trace_batch_id` leaks, trace
POSTs move from `/tracing/ephemeral/batches` to
`/tracing/batches/<leaked-id>/events`, the recorded cassette stops matching, and
the failure lands in whichever unrelated test happens to run later — the symptom
names neither the leak nor the test that caused it.

The autouse `reset_trace_listener_singleton` fixture in the root `conftest.py`
clears the cached instance after every test. These are canaries for that: run on
their own they pass trivially, but under the full suite in random order they fail
if the fixture stops working.
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


def test_listener_setup_flag_starts_clean() -> None:
    """`_listeners_setup` must not survive a test.

    `cleanup_event_handlers` empties the event bus between tests. If the flag
    stayed set, the next listener would consider itself registered and collect
    nothing at all — a silent hole rather than a visible failure.
    """
    assert TraceCollectionListener._listeners_setup is False
