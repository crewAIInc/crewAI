"""Unit tests for the generic interception-hook dispatcher.

These cover the new contract (payload-in/payload-out + HookAborted), the shared
ordered queue between the legacy and new dialects on the four model/tool points,
execution-scoped hooks, fail-open exception handling, telemetry, and the no-op
fast-path overhead budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.hook_events import HookDispatchedEvent
from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear_all,
    dispatch,
    get_hooks,
    on,
    register,
    register_scoped,
    scoped_hooks,
)
from crewai.hooks.llm_hooks import (
    get_before_llm_call_hooks,
    register_before_llm_call_hook,
)
import pytest


@dataclass
class _Ctx:
    payload: object = None
    tool_name: str | None = None
    agent: object = None


@pytest.fixture(autouse=True)
def clear_dispatch_registry():
    """Ensure every test starts and ends with an empty global registry."""
    clear_all()
    yield
    clear_all()


class TestDispatchContract:
    """The core payload-in/payload-out + HookAborted contract."""

    def test_noop_fast_path_returns_context_unchanged(self):
        ctx = _Ctx(payload="original")
        result = dispatch(InterceptionPoint.INPUT, ctx)
        assert result is ctx
        assert ctx.payload == "original"

    def test_return_value_replaces_payload(self):
        def double(ctx):
            return ctx.payload * 2

        register(InterceptionPoint.INPUT, double)
        ctx = _Ctx(payload="ab")
        dispatch(InterceptionPoint.INPUT, ctx)
        assert ctx.payload == "abab"

    def test_in_place_mutation_is_honored(self):
        def mutate(ctx):
            ctx.payload.append(1)
            return None

        register(InterceptionPoint.INPUT, mutate)
        ctx = _Ctx(payload=[])
        dispatch(InterceptionPoint.INPUT, ctx)
        assert ctx.payload == [1]

    def test_hooks_run_in_registration_order(self):
        order: list[int] = []
        register(InterceptionPoint.INPUT, lambda ctx: order.append(1))
        register(InterceptionPoint.INPUT, lambda ctx: order.append(2))
        dispatch(InterceptionPoint.INPUT, _Ctx())
        assert order == [1, 2]

    def test_hook_aborted_propagates_with_reason_and_source(self):
        def blocker(ctx):
            raise HookAborted(reason="nope", source="policy")

        register(InterceptionPoint.INPUT, blocker)
        with pytest.raises(HookAborted) as exc:
            dispatch(InterceptionPoint.INPUT, _Ctx())
        assert exc.value.reason == "nope"
        assert exc.value.source == "policy"

    def test_ordinary_exception_is_swallowed_and_later_hooks_run(self):
        ran: list[str] = []

        def boom(ctx):
            ran.append("boom")
            raise ValueError("bug in user hook")

        def after(ctx):
            ran.append("after")

        register(InterceptionPoint.INPUT, boom)
        register(InterceptionPoint.INPUT, after)
        dispatch(InterceptionPoint.INPUT, _Ctx(), verbose=False)
        assert ran == ["boom", "after"]


class TestOnDecorator:
    """The @on decorator registers and filters like the legacy decorators."""

    def test_on_registers_global_hook(self):
        @on(InterceptionPoint.MEMORY_WRITE)
        def hook(ctx):
            return None

        assert hook in get_hooks(InterceptionPoint.MEMORY_WRITE)

    def test_tool_filter_skips_non_matching_tools(self):
        seen: list[str] = []

        @on(InterceptionPoint.PRE_TOOL_CALL, tools=["allowed_tool"])
        def hook(ctx):
            seen.append(ctx.tool_name)

        dispatch(InterceptionPoint.PRE_TOOL_CALL, _Ctx(tool_name="other_tool"))
        dispatch(InterceptionPoint.PRE_TOOL_CALL, _Ctx(tool_name="allowed_tool"))
        assert seen == ["allowed_tool"]

    def test_agent_filter_skips_non_matching_agents(self):
        seen: list[str] = []

        class _Agent:
            def __init__(self, role):
                self.role = role

        @on(InterceptionPoint.PRE_MODEL_CALL, agents=["Researcher"])
        def hook(ctx):
            seen.append(ctx.agent.role)

        dispatch(InterceptionPoint.PRE_MODEL_CALL, _Ctx(agent=_Agent("Writer")))
        dispatch(InterceptionPoint.PRE_MODEL_CALL, _Ctx(agent=_Agent("Researcher")))
        assert seen == ["Researcher"]


class TestSharedQueueWithLegacyDialect:
    """Legacy registrations and @on hooks compose in one ordered queue."""

    def test_on_and_legacy_share_pre_model_call_queue(self):
        def legacy(ctx):
            return None

        @on(InterceptionPoint.PRE_MODEL_CALL)
        def modern(ctx):
            return None

        register_before_llm_call_hook(legacy)

        queue = get_before_llm_call_hooks()
        assert modern in queue
        assert legacy in queue
        # registration order preserved: modern registered before legacy
        assert queue.index(modern) < queue.index(legacy)


class TestScopedHooks:
    """Execution-scoped hooks run after globals and are discarded on exit."""

    def test_scoped_runs_after_global_then_cleared(self):
        order: list[str] = []
        register(InterceptionPoint.OUTPUT, lambda ctx: order.append("global"))

        with scoped_hooks():
            register_scoped(InterceptionPoint.OUTPUT, lambda ctx: order.append("scoped"))
            dispatch(InterceptionPoint.OUTPUT, _Ctx())

        # outside the scope the scoped hook is gone
        dispatch(InterceptionPoint.OUTPUT, _Ctx())

        assert order == ["global", "scoped", "global"]


class TestTelemetry:
    """dispatch emits a HookDispatchedEvent only when hooks ran."""

    def test_no_event_on_empty_fast_path(self):
        events: list[HookDispatchedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(HookDispatchedEvent)
            def _capture(_source, event):
                events.append(event)

            dispatch(InterceptionPoint.INPUT, _Ctx())

        assert events == []

    def test_event_reports_outcome(self):
        events: list[HookDispatchedEvent] = []

        register(InterceptionPoint.INPUT, lambda ctx: "changed")

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(HookDispatchedEvent)
            def _capture(_source, event):
                events.append(event)

            dispatch(InterceptionPoint.INPUT, _Ctx())

        assert len(events) == 1
        assert events[0].interception_point == "input"
        assert events[0].outcome == "modified"
        assert events[0].hook_count == 1


class TestNoOpOverhead:
    """The no-op fast path must stay cheap (a single dict lookup)."""

    def test_noop_dispatch_overhead_budget(self):
        ctx = _Ctx()
        iterations = 100_000
        start = time.perf_counter()
        for _ in range(iterations):
            dispatch(InterceptionPoint.INPUT, ctx)
        elapsed = time.perf_counter() - start
        # Generous CI-safe budget: < 5µs per no-op dispatch on average.
        assert elapsed / iterations < 5e-6
