"""Conformance suite for the framework-native interception points.

For each wired point this suite asserts the shared contract: the probe hook
sees a well-shaped payload, an in-place/returned modification is honored, and a
:class:`HookAborted` interrupts the step.
"""

from __future__ import annotations

from crewai.flow.flow import Flow, listen, start
from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear_all,
    on,
)
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
