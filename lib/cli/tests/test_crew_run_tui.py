from datetime import datetime
import time

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.observation_events import (
    StepObservationCompletedEvent,
    StepObservationStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai_cli.crew_run_tui import CrewRunApp


def _app_with_plan() -> CrewRunApp:
    app = CrewRunApp()
    app._plan = {
        "plan": "Demo plan",
        "steps": [
            {"step_number": 1, "description": "First"},
            {"step_number": 2, "description": "Second"},
            {"step_number": 3, "description": "Third"},
        ],
    }
    app._plan_step_status = {1: "pending", 2: "pending", 3: "pending"}
    return app


def _log_entry(name: str) -> dict:
    now = time.time()
    return {
        "tool_name": name,
        "status": "success",
        "args": None,
        "result": f"{name} result",
        "error": None,
        "start_time": now,
        "duration": 1.0,
        "task_idx": 1,
    }


def test_plan_step_status_updates_only_the_explicit_step() -> None:
    app = _app_with_plan()

    app._set_plan_step_status(2, "done")

    assert app._plan_step_status == {
        1: "pending",
        2: "done",
        3: "pending",
    }


def test_step_observation_events_update_the_explicit_step() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        future = crewai_event_bus.emit(
            None,
            StepObservationStartedEvent(
                agent_role="Agent",
                step_number=2,
                step_description="Second",
            ),
        )
        if future:
            future.result(timeout=5)

        assert app._plan_step_status == {
            1: "pending",
            2: "active",
            3: "pending",
        }

        future = crewai_event_bus.emit(
            None,
            StepObservationCompletedEvent(
                agent_role="Agent",
                step_number=2,
                step_description="Second",
                step_completed_successfully=True,
            ),
        )
        if future:
            future.result(timeout=5)
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "pending",
        2: "done",
        3: "pending",
    }


def test_tool_usage_events_do_not_advance_plan_steps() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        future = crewai_event_bus.emit(
            None,
            ToolUsageStartedEvent(tool_name="search", tool_args={"query": "CrewAI"}),
        )
        if future:
            future.result(timeout=5)

        now = datetime.now()
        future = crewai_event_bus.emit(
            None,
            ToolUsageFinishedEvent(
                tool_name="search",
                tool_args={"query": "CrewAI"},
                started_at=now,
                finished_at=now,
                output="result",
            ),
        )
        if future:
            future.result(timeout=5)
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "pending",
        2: "pending",
        3: "pending",
    }


def test_streamed_step_observation_updates_named_step_only() -> None:
    app = _app_with_plan()

    updated = app._try_parse_step_observation(
        '{"step_completed_successfully":true,'
        '"key_information_learned":"Step 2 succeeded with the official source."}'
    )

    assert updated is True
    assert app._plan_step_status == {
        1: "pending",
        2: "done",
        3: "pending",
    }


def test_failed_streamed_step_observation_marks_named_step_failed() -> None:
    app = _app_with_plan()

    updated = app._try_parse_step_observation(
        '{"step_completed_successfully":false,'
        '"key_information_learned":"Step 2 failed because the tool failed."}'
    )

    assert updated is True
    assert app._plan_step_status == {
        1: "pending",
        2: "failed",
        3: "pending",
    }


def test_step_observation_json_is_hidden_from_streaming_text() -> None:
    app = _app_with_plan()

    assert (
        app._strip_step_observation_json(
            'Visible before {"step_completed_successfully":true,'
            '"key_information_learned":"Step 2 succeeded."} visible after'
        )
        == "Visible before  visible after"
    )


@pytest.mark.asyncio
async def test_completed_run_keeps_activity_log_keyboard_navigation_active() -> None:
    app = CrewRunApp()

    async with app.run_test(size=(100, 40)) as pilot:
        app._log_entries = [_log_entry("search"), _log_entry("scrape")]

        app._on_crew_done("final output")
        await pilot.pause()

        assert app.focused is app.query_one("#log-panel")

        await pilot.press("down", "enter")
        await pilot.pause()

        assert app._log_cursor == 1
        assert app._log_expanded == {1}

        await pilot.press("up")
        await pilot.pause()

        assert app._log_cursor == 0
