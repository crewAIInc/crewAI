"""Tests for flow-level token usage aggregation

``flow.usage_metrics`` listens to ``LLMCallCompletedEvent`` for the duration
of ``kickoff_async`` so it covers every LLM call inside the flow — crew-led,
tool-led, AND bare ``LLM.call(...)`` from a flow method. We exercise the
aggregator end-to-end through the real event bus with fabricated events and
explicit contextvar control; no live LLM provider is required.
"""

from __future__ import annotations

import contextvars
import os
import tempfile
from typing import Any, Callable
from uuid import uuid4

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
from crewai.flow.async_feedback.types import PendingFeedbackContext
from crewai.flow.flow import Flow, listen, start
from crewai.flow.flow_context import current_flow_id
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.flow.runtime import _usage_dict_to_metrics
from crewai.types.usage_metrics import UsageMetrics


def _emit_llm_call(
    *,
    flow_id: str | None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Emit one fake ``LLMCallCompletedEvent`` with ``current_flow_id`` pinned
    to ``flow_id``.

    Runs in a freshly-copied context so the value the bus snapshots at emit
    time is exactly ``flow_id`` — independent of the calling thread's outer
    context. Mirrors how the real ``LLM.call`` emits events at runtime.
    """
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    for key, value in (
        ("cached_prompt_tokens", cached_prompt_tokens),
        ("reasoning_tokens", reasoning_tokens),
        ("cache_creation_tokens", cache_creation_tokens),
    ):
        if value:
            usage[key] = value
    event = LLMCallCompletedEvent(
        call_id=str(uuid4()),
        model="gpt-4o-mini",
        response="ok",
        call_type=LLMCallType.LLM_CALL,
        usage=usage,
    )

    ctx = contextvars.copy_context()

    def _emit() -> None:
        current_flow_id.set(flow_id)
        future = crewai_event_bus.emit(object(), event)
        if future is not None:
            future.result(timeout=5.0)

    ctx.run(_emit)


class _ScriptedFlow(Flow):
    """A Flow whose ``@start`` delegates to a per-instance ``_script`` closure.

    Each test attaches a script with ``flow._script = lambda f: ...`` so we
    don't redefine a Flow subclass for every scenario.
    """

    @start()
    def run(self) -> None:
        script: Callable[[Flow], None] = getattr(self, "_script", lambda _f: None)
        script(self)


def _run(script: Callable[[Flow], None] = lambda _f: None) -> Flow:
    """Build a ``_ScriptedFlow``, attach ``script``, kickoff. Returns the flow."""
    flow = _ScriptedFlow()
    flow._script = script
    flow.kickoff()
    return flow


class TestUsageDictToMetrics:
    """Unit tests for the dict-to-UsageMetrics normalizer."""

    @pytest.mark.parametrize(
        "usage, expected",
        [
            (None, None),
            ({}, None),
            (
                {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                UsageMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    successful_requests=1,
                ),
            ),
            # total_tokens missing → derived from prompt + completion
            (
                {"prompt_tokens": 4, "completion_tokens": 6},
                UsageMetrics(
                    prompt_tokens=4,
                    completion_tokens=6,
                    total_tokens=10,
                    successful_requests=1,
                ),
            ),
            # Extended provider-specific keys flow through normalization
            (
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "total_tokens": 180,
                    "cached_prompt_tokens": 40,
                    "reasoning_tokens": 25,
                    "cache_creation_tokens": 10,
                },
                UsageMetrics(
                    prompt_tokens=100,
                    completion_tokens=80,
                    total_tokens=180,
                    cached_prompt_tokens=40,
                    reasoning_tokens=25,
                    cache_creation_tokens=10,
                    successful_requests=1,
                ),
            ),
            # Garbage / non-int values coerce to 0 instead of crashing
            (
                {"prompt_tokens": "n/a", "completion_tokens": None, "total_tokens": 7},
                UsageMetrics(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    successful_requests=1,
                ),
            ),
            # Native Anthropic provider emits input_tokens/output_tokens
            (
                {"input_tokens": 12, "output_tokens": 8},
                UsageMetrics(
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                    successful_requests=1,
                ),
            ),
            # Native Gemini provider emits prompt_token_count/candidates_token_count
            (
                {
                    "prompt_token_count": 30,
                    "candidates_token_count": 20,
                    "reasoning_tokens": 5,
                },
                UsageMetrics(
                    prompt_tokens=30,
                    completion_tokens=20,
                    total_tokens=50,
                    reasoning_tokens=5,
                    successful_requests=1,
                ),
            ),
            # OpenAI nests cached_tokens under prompt_tokens_details
            (
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 30},
                },
                UsageMetrics(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    cached_prompt_tokens=30,
                    successful_requests=1,
                ),
            ),
        ],
        ids=[
            "none",
            "empty",
            "all_keys",
            "no_total",
            "extended_keys",
            "garbage",
            "anthropic_aliases",
            "gemini_aliases",
            "openai_nested_cached",
        ],
    )
    def test_normalization(
        self, usage: dict[str, Any] | None, expected: UsageMetrics | None
    ) -> None:
        assert _usage_dict_to_metrics(usage) == expected


class TestFlowUsageAggregation:
    """End-to-end tests driving the listener through the real event bus."""

    def test_sums_every_llm_call_in_the_flow(self) -> None:
        """Multiple LLM calls — including bare ``LLM.call(...)`` made outside
        any crew — accumulate; ``successful_requests`` tracks the call count."""

        def script(flow: Flow) -> None:
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=300, completion_tokens=300)
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=200, completion_tokens=100)
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=20, completion_tokens=20)

        flow = _run(script)

        assert flow.usage_metrics.total_tokens == 940
        assert flow.usage_metrics.prompt_tokens == 520
        assert flow.usage_metrics.completion_tokens == 420
        assert flow.usage_metrics.successful_requests == 3

    def test_returns_zero_when_no_calls_happen(self) -> None:
        flow = _run()
        assert flow.usage_metrics == UsageMetrics()

    def test_ignores_events_from_other_flows(self) -> None:
        """Concurrent flow runs share the singleton bus, so the listener must
        scope itself to its own flow via the contextvar match."""

        def script(flow: Flow) -> None:
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=50, completion_tokens=50)
            _emit_llm_call(flow_id="some-other-flow", prompt_tokens=49_000, completion_tokens=50_999)

        flow = _run(script)

        assert flow.usage_metrics.total_tokens == 100
        assert flow.usage_metrics.successful_requests == 1

    def test_resets_between_kickoffs(self) -> None:
        flow = _ScriptedFlow()
        flow._script = lambda f: _emit_llm_call(
            flow_id=f._flow_match_id, prompt_tokens=250, completion_tokens=250
        )

        flow.kickoff()
        flow.kickoff()

        assert flow.usage_metrics.total_tokens == 500
        assert flow.usage_metrics.successful_requests == 1

    def test_usage_metrics_returns_independent_copy(self) -> None:
        """``usage_metrics`` must return a copy, not the internal instance —
        otherwise callers can clobber the in-flight accumulator."""

        flow = _run(
            lambda f: _emit_llm_call(
                flow_id=f._flow_match_id, prompt_tokens=50, completion_tokens=50
            )
        )

        snapshot = flow.usage_metrics
        snapshot.total_tokens = 999_999

        assert flow.usage_metrics.total_tokens == 100

    def test_handler_is_unregistered_after_kickoff(self) -> None:
        """Long-lived workers (Celery, devkit) must not leak one handler per
        kickoff on the singleton bus, on either the success or failure path."""

        def handler_count() -> int:
            return len(
                crewai_event_bus._sync_handlers.get(LLMCallCompletedEvent, frozenset())
            )

        before = handler_count()

        flow = _ScriptedFlow()
        flow._script = lambda f: _emit_llm_call(
            flow_id=f._flow_match_id, prompt_tokens=5, completion_tokens=5
        )
        for _ in range(3):
            flow.kickoff()

        assert handler_count() == before

        def boom(_f: Flow) -> None:
            raise RuntimeError("boom")

        failing = _ScriptedFlow()
        failing._script = boom

        with pytest.raises(RuntimeError, match="boom"):
            failing.kickoff()

        assert handler_count() == before

    def test_kickoff_flushes_event_bus_before_returning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`kickoff_async` must drain pending LLMCallCompletedEvent handlers
        before detaching the listener — otherwise late handlers landing on
        the threadpool would be lost on short flows. Mirrors the flush
        ``Crew.kickoff()`` performs before reporting ``token_usage``."""

        flush_calls: list[None] = []
        original_flush = crewai_event_bus.flush

        def tracked_flush(*args: Any, **kwargs: Any) -> bool:
            flush_calls.append(None)
            return original_flush(*args, **kwargs)

        monkeypatch.setattr(crewai_event_bus, "flush", tracked_flush)

        flow = _ScriptedFlow()
        flow._script = lambda f: _emit_llm_call(
            flow_id=f._flow_match_id, prompt_tokens=3, completion_tokens=4
        )
        flow.kickoff()

        assert flush_calls, "kickoff did not flush the event bus before returning"
        assert flow.usage_metrics.total_tokens == 7

    def test_stale_handler_from_prior_kickoff_does_not_contaminate(self) -> None:
        """A handler still queued from a prior kickoff must not write into
        a later kickoff's accumulator. The handler's closure captures its
        own accumulator object, so any late writes land on an orphaned
        instance and the live ``usage_metrics`` is unaffected."""

        captured: dict[str, Any] = {}

        def script(flow: Flow) -> None:
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=10, completion_tokens=10)
            captured["handler"] = flow._usage_aggregation_handler
            captured["match_id"] = flow._flow_match_id

        flow = _run(script)
        assert flow.usage_metrics.total_tokens == 20

        flow._script = lambda f: None
        flow.kickoff()
        assert flow.usage_metrics.total_tokens == 0

        stale_handler = captured["handler"]
        assert stale_handler is not None

        stale_event = LLMCallCompletedEvent(
            call_id=str(uuid4()),
            model="gpt-4o-mini",
            response="ok",
            call_type=LLMCallType.LLM_CALL,
            usage={"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 1998},
        )
        ctx = contextvars.copy_context()
        ctx.run(lambda: (current_flow_id.set(captured["match_id"]), stale_handler(object(), stale_event)))

        assert flow.usage_metrics.total_tokens == 0

    def test_pause_detaches_listener_and_does_not_leak(self) -> None:
        """When ``kickoff_async`` pauses for human feedback, the listener
        must be detached from the singleton bus to avoid leaking handlers
        across abandoned paused instances. Pre-pause LLM events still
        count because the bus snapshots handlers at emit time. Late
        events emitted after the pause returns do not count for this
        instance — resume paths re-attach a fresh listener."""

        from crewai.flow.async_feedback.types import HumanFeedbackPending

        captured: dict[str, Any] = {}

        class _PausingFlow(Flow):
            @start()
            def begin(self) -> None:
                _emit_llm_call(
                    flow_id=self._flow_match_id,
                    prompt_tokens=10,
                    completion_tokens=20,
                )
                captured["pre_pause_total"] = self.usage_metrics.total_tokens
                raise HumanFeedbackPending(
                    context=PendingFeedbackContext(
                        flow_id=self.flow_id,
                        flow_class="_PausingFlow",
                        method_name="begin",
                        method_output="content",
                        message="Review:",
                    )
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "f.db"))
            flow = _PausingFlow(persistence=persistence)
            result = flow.kickoff()

            assert isinstance(result, HumanFeedbackPending)
            assert captured["pre_pause_total"] == 30
            assert flow._usage_aggregation_handler is None

            # A late event emitted after the pause does not reach the
            # detached listener, so the running total is unchanged.
            _emit_llm_call(
                flow_id=flow._flow_match_id,
                prompt_tokens=2,
                completion_tokens=3,
            )
            assert flow.usage_metrics.total_tokens == 30

    def test_aggregates_resume_after_from_pending(self) -> None:
        """A flow restored via ``from_pending`` is a fresh instance with no
        ``_flow_match_id``; without seeding it, the listener attached in
        ``resume_async`` either ignores its own LLM calls or absorbs unrelated
        ones. ``from_pending`` must seed the match id so the resume-phase
        aggregator counts our own calls and only our own calls."""

        class _ResumeFlow(Flow):
            @start()
            def begin(self) -> str:
                return "content"

            @listen(begin)
            def on_begin(self, _feedback: Any) -> str:
                _emit_llm_call(
                    flow_id=self._flow_match_id,
                    prompt_tokens=100,
                    completion_tokens=50,
                )
                _emit_llm_call(
                    flow_id="some-other-flow",
                    prompt_tokens=9_999,
                    completion_tokens=9_999,
                )
                return "done"

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "f.db"))
            flow_id = "usage-resume-test"
            persistence.save_pending_feedback(
                flow_uuid=flow_id,
                context=PendingFeedbackContext(
                    flow_id=flow_id,
                    flow_class="_ResumeFlow",
                    method_name="begin",
                    method_output="content",
                    message="Review:",
                ),
                state_data={"id": flow_id},
            )

            flow = _ResumeFlow.from_pending(flow_id, persistence)
            assert flow._flow_match_id == flow.flow_id

            flow.resume("ok")

            assert flow.usage_metrics.total_tokens == 150
            assert flow.usage_metrics.prompt_tokens == 100
            assert flow.usage_metrics.completion_tokens == 50
            assert flow.usage_metrics.successful_requests == 1

    def test_resume_aggregates_under_foreign_flow_context(self) -> None:
        """Resume must override an already-set ``current_flow_id`` so its
        own LLM events match the listener's filter even when invoked from
        inside another flow's active context."""

        class _ResumeFlow(Flow):
            @start()
            def begin(self) -> str:
                return "content"

            @listen(begin)
            def on_begin(self, _feedback: Any) -> str:
                _emit_llm_call(
                    flow_id=self._flow_match_id,
                    prompt_tokens=42,
                    completion_tokens=8,
                )
                return "done"

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "f.db"))
            flow_id = "resume-foreign-context"
            persistence.save_pending_feedback(
                flow_uuid=flow_id,
                context=PendingFeedbackContext(
                    flow_id=flow_id,
                    flow_class="_ResumeFlow",
                    method_name="begin",
                    method_output="content",
                    message="Review:",
                ),
                state_data={"id": flow_id},
            )

            foreign_token = current_flow_id.set("some-parent-flow")
            try:
                flow = _ResumeFlow.from_pending(flow_id, persistence)
                flow.resume("ok")
            finally:
                current_flow_id.reset(foreign_token)

            assert flow.usage_metrics.total_tokens == 50
            assert flow.usage_metrics.successful_requests == 1
