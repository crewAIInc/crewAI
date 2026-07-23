"""Conformance suite for the framework-native interception points.

For each wired point this suite asserts the shared contract: the probe hook
sees a well-shaped payload, an in-place/returned modification is honored, and a
:class:`HookAborted` interrupts the step.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.crew_events import CrewKickoffCompletedEvent
from crewai.flow.flow import Flow, listen, start
from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear_all,
    on,
)
from crewai.task import Task
import pytest


@pytest.fixture(autouse=True)
def clear_dispatch_registry():
    clear_all()
    yield
    clear_all()


class _SimpleFlow(Flow):
    @start()
    def begin(self):
        return "begin"

    @listen(begin)
    def finish(self, _):
        return "flow-result"


class _FailingFlow(Flow):
    @start()
    def begin(self):
        raise RuntimeError("flow boom")


class _ReentrantFailingFlow(Flow):
    """Kicks itself off once from inside a method, then fails in the outer run."""

    @start()
    async def begin(self):
        if getattr(self, "_reentered", False):
            return "inner-ok"
        self._reentered = True
        await self.kickoff_async()
        raise RuntimeError("outer boom")


class TestFlowExecutionBoundaries:
    """execution_start / input / output / execution_end on a flow."""

    def test_all_boundary_points_fire_once(self):
        fired: list[str] = []

        for point in (
            InterceptionPoint.EXECUTION_START,
            InterceptionPoint.INPUT,
            InterceptionPoint.OUTPUT,
            InterceptionPoint.EXECUTION_END,
        ):

            @on(point)
            def _probe(ctx, _point=point):
                fired.append(_point.value)

        _SimpleFlow().kickoff(inputs={"seed": 1})

        assert fired == [
            "execution_start",
            "input",
            "output",
            "execution_end",
        ]

    def test_output_modification_is_honored(self):
        @on(InterceptionPoint.OUTPUT)
        def rewrite(ctx):
            return "intercepted"

        result = _SimpleFlow().kickoff()
        assert result == "intercepted"

    def test_input_payload_carries_inputs(self):
        seen: dict = {}

        @on(InterceptionPoint.INPUT)
        def capture(ctx):
            seen.update(ctx.payload or {})

        _SimpleFlow().kickoff(inputs={"seed": 42})
        assert seen == {"seed": 42}

    def test_abort_at_execution_start_interrupts(self):
        @on(InterceptionPoint.EXECUTION_START)
        def block(ctx):
            raise HookAborted(reason="not allowed", source="policy")

        with pytest.raises(HookAborted) as exc:
            _SimpleFlow().kickoff()
        assert exc.value.reason == "not allowed"


class TestFlowStepPoints:
    """pre_step / post_step for flow methods (kind=flow_method)."""

    def test_pre_and_post_step_fire_per_method(self):
        kinds: list[tuple[str, str | None]] = []

        @on(InterceptionPoint.PRE_STEP)
        def pre(ctx):
            kinds.append(("pre", ctx.step_name))

        @on(InterceptionPoint.POST_STEP)
        def post(ctx):
            kinds.append(("post", ctx.step_name))

        _SimpleFlow().kickoff()

        assert ("pre", "begin") in kinds
        assert ("post", "begin") in kinds
        assert ("pre", "finish") in kinds
        assert ("post", "finish") in kinds

    def test_post_step_can_rewrite_method_output(self):
        @on(InterceptionPoint.POST_STEP)
        def rewrite(ctx):
            if ctx.step_name == "finish":
                return "rewritten"
            return None

        assert _SimpleFlow().kickoff() == "rewritten"


class TestTaskStepPoints:
    """pre_step / post_step for task execution (kind=task)."""

    def test_post_step_rewrite_is_persisted_to_output_file(
        self, tmp_path, monkeypatch
    ):
        @on(InterceptionPoint.POST_STEP)
        def sanitize(ctx):
            return ctx.payload.model_copy(update={"raw": "sanitized output"})

        monkeypatch.chdir(tmp_path)
        agent = Agent(role="Writer", goal="Write", backstory="Writes things.")
        task = Task(
            description="Write something",
            expected_output="Some text",
            output_file="output.txt",
            agent=agent,
        )

        with patch.object(Agent, "execute_task", return_value="original output"):
            result = task.execute_sync(agent=agent)

        assert result.raw == "sanitized output"
        assert (tmp_path / "output.txt").read_text() == "sanitized output"


class TestExecutionEndOnFailure:
    """execution_end fires exactly once, on success and on failure alike."""

    @staticmethod
    def _crew() -> Crew:
        agent = Agent(role="Writer", goal="Write", backstory="Writes things.")
        task = Task(
            description="Write something",
            expected_output="Some text",
            agent=agent,
        )
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def test_crew_success_fires_completed_once(self):
        seen: list[tuple[str, BaseException | None]] = []

        @on(InterceptionPoint.EXECUTION_END)
        def capture(ctx):
            seen.append((ctx.status, ctx.error))

        with patch.object(Agent, "execute_task", return_value="fine"):
            self._crew().kickoff()

        assert seen == [("completed", None)]

    def test_crew_failure_fires_failed_once_and_reraises(self):
        seen = []

        @on(InterceptionPoint.EXECUTION_END)
        def capture(ctx):
            seen.append(ctx)

        error = RuntimeError("crew boom")
        with patch.object(Agent, "execute_task", side_effect=error):
            with pytest.raises(RuntimeError, match="crew boom"):
                self._crew().kickoff()

        assert len(seen) == 1
        assert seen[0].status == "failed"
        assert seen[0].error is error
        assert seen[0].output is None

    def test_crew_kickoff_async_failure_fires_failed_once(self):
        seen = []

        @on(InterceptionPoint.EXECUTION_END)
        def capture(ctx):
            seen.append(ctx)

        with patch.object(
            Agent, "execute_task", side_effect=RuntimeError("crew boom")
        ):
            with pytest.raises(RuntimeError, match="crew boom"):
                asyncio.run(self._crew().kickoff_async())

        assert len(seen) == 1
        assert seen[0].status == "failed"

    def test_flow_success_fires_completed_once(self):
        seen: list[tuple[str, BaseException | None]] = []

        @on(InterceptionPoint.EXECUTION_END)
        def capture(ctx):
            seen.append((ctx.status, ctx.error))

        _SimpleFlow().kickoff()

        assert seen == [("completed", None)]

    def test_flow_failure_fires_failed_once_and_reraises(self):
        seen = []

        @on(InterceptionPoint.EXECUTION_END)
        def capture(ctx):
            seen.append(ctx)

        with pytest.raises(RuntimeError, match="flow boom"):
            _FailingFlow().kickoff()

        assert len(seen) == 1
        assert seen[0].status == "failed"
        assert isinstance(seen[0].error, RuntimeError)
        assert seen[0].output is None

    def test_reentrant_flow_kickoff_pairs_ends_per_invocation(self):
        seen: list[tuple[str, str | None]] = []

        @on(InterceptionPoint.EXECUTION_START)
        def capture_start(ctx):
            seen.append(("start", None))

        @on(InterceptionPoint.EXECUTION_END)
        def capture_end(ctx):
            seen.append(("end", ctx.status))

        with pytest.raises(RuntimeError, match="outer boom"):
            _ReentrantFailingFlow().kickoff()

        assert seen == [
            ("start", None),
            ("start", None),
            ("end", "completed"),
            ("end", "failed"),
        ]

    def test_no_execution_end_when_execution_start_aborts(self):
        seen = []

        @on(InterceptionPoint.EXECUTION_START)
        def block(ctx):
            raise HookAborted(reason="blocked")

        @on(InterceptionPoint.EXECUTION_END)
        def capture(ctx):
            seen.append(ctx)

        with pytest.raises(HookAborted):
            _SimpleFlow().kickoff()
        with pytest.raises(HookAborted):
            self._crew().kickoff()

        assert seen == []

    def test_aborting_execution_end_hook_fires_once_for_flow(self):
        calls: list[str] = []

        @on(InterceptionPoint.EXECUTION_END)
        def abort_end(ctx):
            calls.append(ctx.status)
            raise HookAborted(reason="no")

        with pytest.raises(HookAborted):
            _SimpleFlow().kickoff()

        assert calls == ["completed"]

    def test_aborting_execution_end_hook_fires_once_for_crew(self):
        calls: list[str] = []

        @on(InterceptionPoint.EXECUTION_END)
        def abort_end(ctx):
            calls.append(ctx.status)
            raise HookAborted(reason="no")

        with patch.object(Agent, "execute_task", return_value="fine"):
            with pytest.raises(HookAborted):
                self._crew().kickoff()

        assert calls == ["completed"]


class TestCrewOutput:
    def test_output_modification_reaches_kickoff_completed_event(self):
        @on(InterceptionPoint.OUTPUT)
        def append_notice(ctx):
            if hasattr(ctx.payload, "raw") and isinstance(ctx.payload.raw, str):
                ctx.payload.raw += "\nchanged by hook"
            return None

        completed_raw: list[str] = []

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def capture_completed(_source, event: CrewKickoffCompletedEvent):
            completed_raw.append(event.output.raw)

        agent = Agent(role="Writer", goal="Write", backstory="Writes things.")
        task = Task(
            description="Write something",
            expected_output="Some text",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        with patch.object(Agent, "execute_task", return_value="original output"):
            result = crew.kickoff()
        crewai_event_bus.flush()

        assert result.raw.endswith("changed by hook")
        assert completed_raw
        assert completed_raw[-1].endswith("changed by hook")
