from __future__ import annotations

import time

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import MethodExecutionStartedEvent


def test_flow_method_events_run_sync_handlers_before_emit_returns() -> None:
    handled: list[str] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def record_method_start(_source: object, _event: MethodExecutionStartedEvent) -> None:
            time.sleep(0.05)
            handled.append("method_start")

        future = crewai_event_bus.emit(
            object(),
            MethodExecutionStartedEvent(
                flow_name="flow",
                method_name="step_one",
                state={},
            ),
        )

        assert handled == ["method_start"]
        assert future is None
