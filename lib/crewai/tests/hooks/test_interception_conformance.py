"""Conformance suite for the framework-native interception points.

For each wired point this suite asserts the shared contract: the probe hook
sees a well-shaped payload, an in-place/returned modification is honored, and a
:class:`HookAborted` interrupts the step.
"""

from __future__ import annotations

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
