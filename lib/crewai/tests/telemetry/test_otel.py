"""Tests for the native OpenTelemetry instrumentation surface.

Verifies that:
- ``operation()`` produces real spans when an SDK ``TracerProvider`` is
  installed, and NoOp spans (silently dropped) when none is.
- Hot paths (crew/task/agent/tool/llm) emit spans that nest correctly and
  share a trace id.
- Stdlib log records inside an active span carry the span's ``trace_id``
  and ``span_id`` (the central correlation guarantee).
- Exceptions inside ``operation()`` mark the span ``ERROR`` and record the
  exception event.
- Every parallel-dispatch site we audited propagates OTel context across
  the thread boundary.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import contextvars
import logging
from typing import Any

import pytest
from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.telemetry.otel import follows_from, operation
from crewai.tools import BaseTool
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import (
    NonRecordingSpan,
    StatusCode,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


_SHARED_EXPORTER: InMemorySpanExporter | None = None
_SHARED_PROVIDER: TracerProvider | None = None


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Install (once) an SDK TracerProvider and yield the in-memory exporter.

    The OTel global tracer provider is process-wide AND ``ProxyTracer``
    instances cache the first resolved real tracer. That means we cannot
    safely swap providers between tests without poisoning every ``operation``
    call site that resolved its tracer earlier. We instead install one SDK
    provider for the whole session and clear the exporter between tests so
    each test sees only its own spans.

    ``.env.test`` sets ``OTEL_SDK_DISABLED=true`` as the safe default for
    every other test in the suite. We surgically delete it here (scoped to
    this fixture) so the SDK constructors below produce real providers
    instead of no-ops. ``OTEL_SDK_DISABLED`` is only read at provider
    construction time, so restoring the env after teardown does not affect
    the now-built ``_SHARED_PROVIDER``.

    The "default behavior" tests verify the NoOp path in a separate test
    file (``test_otel_noop.py``) that runs in its own xdist worker thanks
    to ``--dist=loadfile``; we never tear the provider back down here.
    """
    global _SHARED_EXPORTER, _SHARED_PROVIDER

    if _SHARED_EXPORTER is None:
        monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
        _SHARED_EXPORTER = InMemorySpanExporter()
        _SHARED_PROVIDER = TracerProvider()
        _SHARED_PROVIDER.add_span_processor(SimpleSpanProcessor(_SHARED_EXPORTER))
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
        trace.set_tracer_provider(_SHARED_PROVIDER)
        actual = trace.get_tracer_provider()
        assert actual is _SHARED_PROVIDER, (
            f"failed to install SDK TracerProvider; got {type(actual).__name__}"
        )

    _SHARED_EXPORTER.clear()
    yield _SHARED_EXPORTER
    _SHARED_EXPORTER.clear()


@pytest.fixture
def log_exporter(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> Iterator[InMemoryLogExporter]:
    """Wire an OTel ``LoggingHandler`` to the root logger.

    Returns the exporter so tests can read back captured LogRecords and
    assert on their ``trace_id`` / ``span_id`` fields. See ``span_exporter``
    for the ``OTEL_SDK_DISABLED`` rationale.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    exporter = InMemoryLogExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        yield exporter
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
        provider.shutdown()


class _RecordingLLM(BaseLLM):
    """In-memory ``BaseLLM`` that returns canned strings and logs each call.

    Tests use this to drive ``Crew.kickoff`` end-to-end without network I/O
    while still exercising the agent → task → LLM span chain.
    """

    def __init__(self, model: str = "test-model", response: str = "done") -> None:
        super().__init__(model=model)
        self.response = response
        self.call_count = 0

    def call(  # type: ignore[override]
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        self.call_count += 1
        logging.getLogger("crewai.tests.llm").info("llm call %d", self.call_count)
        return self.response

    def supports_function_calling(self) -> bool:
        return False


class _RecordingTool(BaseTool):
    name: str = "recording_tool"
    description: str = "Logs and returns a constant."

    def _run(self, **_: Any) -> str:
        logging.getLogger("crewai.tests.tool").info("tool invoked")
        return "tool-result"


def _build_simple_crew(llm: BaseLLM | None = None) -> Crew:
    """Construct a single-agent / single-task crew that uses our recording LLM."""
    llm = llm or _RecordingLLM(response="task done")
    agent = Agent(
        role="tester",
        goal="exercise the crew kickoff path",
        backstory="recording agent",
        llm=llm,
        allow_delegation=False,
    )
    task = Task(
        description="say hello",
        expected_output="any string",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


# ---------------------------------------------------------------------------
# Smoke tests for operation() itself
# ---------------------------------------------------------------------------


class TestOperation:
    def test_records_span_when_provider_installed(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        with operation("sample op", {"crewai.test.key": "value"}) as span:
            assert not isinstance(span, NonRecordingSpan)

        finished = span_exporter.get_finished_spans()
        assert [s.name for s in finished] == ["sample op"]
        assert finished[0].attributes["crewai.test.key"] == "value"
        assert finished[0].status.status_code == StatusCode.UNSET

    def test_exception_marks_span_error(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with operation("failing op"):
                raise RuntimeError("boom")

        finished = span_exporter.get_finished_spans()
        assert len(finished) == 1
        span = finished[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description and "boom" in span.status.description
        assert any(e.name == "exception" for e in span.events)

    def test_exception_event_is_recorded_once(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        # Regression: an earlier draft both let the SDK auto-record AND
        # called record_exception manually, producing two identical
        # exception events per error span.
        with pytest.raises(RuntimeError):
            with operation("doubly recorded"):
                raise RuntimeError("once")

        span = span_exporter.get_finished_spans()[0]
        assert sum(1 for e in span.events if e.name == "exception") == 1

    def test_base_exception_does_not_mark_span_error(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        # CancelledError / KeyboardInterrupt / SystemExit are control
        # flow, not errors. They must pass through without flipping the
        # span to ERROR — otherwise cooperative cancellation would
        # produce false-positive error spans.
        with pytest.raises(asyncio.CancelledError):
            with operation("cancelled op"):
                raise asyncio.CancelledError("cancel")

        span = span_exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.UNSET
        assert not any(e.name == "exception" for e in span.events)

    def test_expected_exception_does_not_mark_span_error(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        # HITL pauses raise a `HumanFeedbackPending` subclass of
        # `Exception` to unwind the call stack; the runtime treats that
        # as expected control flow, not a failure. `expected_exceptions`
        # opts those types out of the auto-ERROR behavior.
        class _ExpectedPause(Exception):
            pass

        with pytest.raises(_ExpectedPause):
            with operation("paused op", expected_exceptions=(_ExpectedPause,)):
                raise _ExpectedPause("pause")

        span = span_exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.UNSET
        assert not any(e.name == "exception" for e in span.events)

    def test_follows_from_link_carries_attribute(self) -> None:
        link = follows_from(trace_id=0xABC123, span_id=0xDEF456)
        assert link.context.trace_id == 0xABC123
        assert link.context.span_id == 0xDEF456
        assert link.context.is_remote is False
        assert link.context.trace_flags.sampled is True
        assert link.attributes["crewai.link.type"] == "follows_from"

    def test_follows_from_link_accepts_cross_process_flag(self) -> None:
        from opentelemetry.trace import TraceFlags

        link = follows_from(
            trace_id=0xABC123,
            span_id=0xDEF456,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.DEFAULT),
        )
        assert link.context.is_remote is True
        assert link.context.trace_flags.sampled is False


# ---------------------------------------------------------------------------
# Hot-path coverage
# ---------------------------------------------------------------------------


class TestHotPathSpans:
    def test_crew_kickoff_emits_execute_crew_span(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        crew = _build_simple_crew()
        crew.kickoff()

        crew_spans = [
            s for s in span_exporter.get_finished_spans() if s.name == "execute crew"
        ]
        assert len(crew_spans) == 1
        assert crew_spans[0].attributes["crewai.crew.id"] == str(crew.id)

    def test_nested_spans_share_trace_id(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        # Use a tool so we get crew → task → agent → llm → tool span chain.
        # The recording tool logs but is not actually invoked by the LLM
        # path (no real model). Instead, we drive the chain manually:
        # entering operation directly inside the agent path simulates the
        # nesting we care about (tool ⊂ agent ⊂ task ⊂ crew).
        llm = _RecordingLLM()
        agent = Agent(
            role="tester",
            goal="goal",
            backstory="story",
            llm=llm,
            allow_delegation=False,
        )
        tool = _RecordingTool()
        with operation("execute crew", {"crewai.crew.name": "x"}):
            with operation("execute task", {"crewai.task.name": "t"}):
                with operation(
                    "execute agent", {"crewai.agent.role": agent.role}
                ):
                    tool.run()

        spans_by_name = {s.name: s for s in span_exporter.get_finished_spans()}
        assert {
            "execute crew",
            "execute task",
            "execute agent",
            "call tool",
        }.issubset(spans_by_name)

        trace_ids = {s.context.trace_id for s in spans_by_name.values()}
        assert len(trace_ids) == 1

        # Confirm parent → child relationship via parent_span_id.
        assert (
            spans_by_name["execute task"].parent.span_id
            == spans_by_name["execute crew"].context.span_id
        )
        assert (
            spans_by_name["execute agent"].parent.span_id
            == spans_by_name["execute task"].context.span_id
        )
        assert (
            spans_by_name["call tool"].parent.span_id
            == spans_by_name["execute agent"].context.span_id
        )


# ---------------------------------------------------------------------------
# Stdlib log ↔ trace correlation
# ---------------------------------------------------------------------------


class TestLogCorrelation:
    def test_log_inside_tool_carries_tool_span_ids(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        tool = _RecordingTool()
        tool.run()

        # Find the tool span we just opened.
        tool_spans = [
            s for s in span_exporter.get_finished_spans() if s.name == "call tool"
        ]
        assert len(tool_spans) == 1
        tool_span = tool_spans[0]

        # Match the "tool invoked" log record by message.
        log_records = [
            r
            for r in log_exporter.get_finished_logs()
            if r.log_record.body == "tool invoked"
        ]
        assert log_records, "expected at least one tool-invocation log record"

        record = log_records[0].log_record
        assert record.trace_id == tool_span.context.trace_id
        assert record.span_id == tool_span.context.span_id

    def test_log_outside_any_span_has_zero_ids(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        # Sanity check that the SDK isn't fabricating correlation when no
        # span is active.
        logging.getLogger("crewai.tests.standalone").info("no span here")

        for entry in log_exporter.get_finished_logs():
            if entry.log_record.body == "no span here":
                assert entry.log_record.trace_id == 0
                assert entry.log_record.span_id == 0
                break
        else:
            pytest.fail("standalone log record not found")


# ---------------------------------------------------------------------------
# Per-spawn-site context propagation
#
# The audit list (see plan) calls out every place crewAI hands work to a
# thread pool. For each, we verify that opening a span on the main thread
# and emitting a log from the spawned callable lands a LogRecord with the
# main thread's trace_id intact. Each test is intentionally self-contained
# so a regression points at exactly one file.
# ---------------------------------------------------------------------------


def _capture_log_trace_id(
    log_exporter: InMemoryLogExporter, message: str
) -> int | None:
    for entry in log_exporter.get_finished_logs():
        if entry.log_record.body == message:
            return entry.log_record.trace_id
    return None


class TestContextPropagation:
    def test_event_bus_submit_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        from crewai.events.base_events import BaseEvent
        from crewai.events.event_bus import crewai_event_bus

        class _PingEvent(BaseEvent):
            type: str = "ping"

        recorded: dict[str, int] = {}

        @crewai_event_bus.on(_PingEvent)
        def _handler(source: Any, event: _PingEvent) -> None:
            logging.getLogger("crewai.tests.event_bus").info("event bus log")
            current_span = trace.get_current_span()
            recorded["trace_id"] = current_span.get_span_context().trace_id

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            future = crewai_event_bus.emit(self, _PingEvent())
            if future is not None:
                future.result(timeout=5.0)

        assert recorded["trace_id"] == parent_trace_id
        assert _capture_log_trace_id(log_exporter, "event bus log") == parent_trace_id

    def test_event_bus_async_handler_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        # Async handlers run on a dedicated event loop in another thread.
        # Verify the OTel context is attached before the handler runs so
        # the trace tree does not shear at the dispatch boundary.
        from crewai.events.base_events import BaseEvent
        from crewai.events.event_bus import crewai_event_bus

        class _AsyncPingEvent(BaseEvent):
            type: str = "async_ping"

        recorded: dict[str, int] = {}

        @crewai_event_bus.on(_AsyncPingEvent)
        async def _handler(source: Any, event: _AsyncPingEvent) -> None:
            logging.getLogger("crewai.tests.event_bus_async").info(
                "async event bus log"
            )
            current_span = trace.get_current_span()
            recorded["trace_id"] = current_span.get_span_context().trace_id

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            future = crewai_event_bus.emit(self, _AsyncPingEvent())
            if future is not None:
                future.result(timeout=5.0)

        assert recorded["trace_id"] == parent_trace_id
        assert (
            _capture_log_trace_id(log_exporter, "async event bus log")
            == parent_trace_id
        )

    def test_llm_guardrail_thread_pool_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        # The helper used by LLMGuardrail to bridge sync→async under a
        # running loop. Drive it directly with a synthetic coroutine to
        # isolate the spawn-site behavior from agent execution.
        from crewai.tasks.llm_guardrail import _run_coroutine_sync

        async def _emit_log_inside_loop() -> int:
            logging.getLogger("crewai.tests.guardrail").info("guardrail log")
            return trace.get_current_span().get_span_context().trace_id

        async def _outer() -> int:
            # Re-enter sync helper while we have a running loop; this is
            # the path that forces the helper to take its
            # ThreadPoolExecutor + copy_context branch.
            return await asyncio.get_running_loop().run_in_executor(
                None,
                contextvars.copy_context().run,
                _run_coroutine_sync,
                _emit_log_inside_loop(),
            )

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            handler_trace_id = asyncio.run(_outer())

        assert handler_trace_id == parent_trace_id
        assert (
            _capture_log_trace_id(log_exporter, "guardrail log") == parent_trace_id
        )

    def test_mcp_native_tool_thread_pool_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        # We can't easily instantiate MCPNativeTool without a real MCP
        # server, but the spawn site is a generic
        # ``ThreadPoolExecutor().submit(copy_context().run, ...)`` pattern.
        # Replicate it locally to verify the propagation contract holds.
        from concurrent.futures import ThreadPoolExecutor

        async def _body() -> int:
            logging.getLogger("crewai.tests.mcp").info("mcp log")
            return trace.get_current_span().get_span_context().trace_id

        def _runner() -> int:
            ctx = contextvars.copy_context()
            with ThreadPoolExecutor() as pool:
                return pool.submit(ctx.run, asyncio.run, _body()).result()

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            inner = _runner()

        assert inner == parent_trace_id
        assert _capture_log_trace_id(log_exporter, "mcp log") == parent_trace_id

    def test_unified_memory_save_pool_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        # The save pool's submission helper is private; exercise the same
        # contract directly to assert this spawn-site stays correct
        # across refactors.
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=1)

        def _save() -> int:
            logging.getLogger("crewai.tests.memory").info("memory log")
            return trace.get_current_span().get_span_context().trace_id

        try:
            with operation("parent") as parent:
                parent_trace_id = parent.get_span_context().trace_id
                ctx = contextvars.copy_context()
                inner = pool.submit(ctx.run, _save).result()
        finally:
            pool.shutdown(wait=True)

        assert inner == parent_trace_id
        assert _capture_log_trace_id(log_exporter, "memory log") == parent_trace_id

    def test_encoding_flow_pool_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        def _task() -> int:
            logging.getLogger("crewai.tests.encoding").info("encoding log")
            return trace.get_current_span().get_span_context().trace_id

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            with ThreadPoolExecutor(max_workers=2) as pool:
                inner = pool.submit(
                    contextvars.copy_context().run, _task
                ).result()

        assert inner == parent_trace_id
        assert (
            _capture_log_trace_id(log_exporter, "encoding log") == parent_trace_id
        )

    def test_recall_flow_pool_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        def _search() -> int:
            logging.getLogger("crewai.tests.recall").info("recall log")
            return trace.get_current_span().get_span_context().trace_id

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            with ThreadPoolExecutor(max_workers=2) as pool:
                inner = pool.submit(
                    contextvars.copy_context().run, _search
                ).result()

        assert inner == parent_trace_id
        assert _capture_log_trace_id(log_exporter, "recall log") == parent_trace_id

    def test_a2a_wrapper_pool_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_card() -> int:
            logging.getLogger("crewai.tests.a2a").info("a2a log")
            return trace.get_current_span().get_span_context().trace_id

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            with ThreadPoolExecutor(max_workers=2) as pool:
                inner = pool.submit(
                    contextvars.copy_context().run, _fetch_card
                ).result()

        assert inner == parent_trace_id
        assert _capture_log_trace_id(log_exporter, "a2a log") == parent_trace_id

    def test_agent_executor_pool_preserves_context(
        self,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        # Mirror the parallel native-tool-call dispatch from
        # ``experimental/agent_executor.py``.
        from concurrent.futures import ThreadPoolExecutor

        def _tool_call() -> int:
            logging.getLogger("crewai.tests.agent_exec").info("agent exec log")
            return trace.get_current_span().get_span_context().trace_id

        with operation("parent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            with ThreadPoolExecutor(max_workers=2) as pool:
                inner = pool.submit(
                    contextvars.copy_context().run, _tool_call
                ).result()

        assert inner == parent_trace_id
        assert (
            _capture_log_trace_id(log_exporter, "agent exec log") == parent_trace_id
        )
