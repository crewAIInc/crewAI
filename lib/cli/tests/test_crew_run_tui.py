from datetime import datetime
import time

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.observation_events import (
    GoalAchievedEarlyEvent,
    PlanRefinementEvent,
    PlanReplanTriggeredEvent,
    PlanStepCompletedEvent,
    PlanStepStartedEvent,
    StepObservationCompletedEvent,
    StepObservationFailedEvent,
    StepObservationStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai_cli import run_crew
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


def _emit_event(event: object) -> None:
    future = crewai_event_bus.emit(None, event)
    if future:
        future.result(timeout=5)


def test_chain_deploy_skips_validation_after_auth_retry(monkeypatch) -> None:
    create_calls: list[dict[str, object]] = []
    login_calls: list[bool] = []

    class FakeDeployCommand:
        attempts = 0

        def create_crew(self, **kwargs) -> None:
            create_calls.append(kwargs)
            FakeDeployCommand.attempts += 1
            if FakeDeployCommand.attempts == 1:
                raise SystemExit(1)

    class FakeAuthenticationCommand:
        def login(self) -> None:
            login_calls.append(True)

    monkeypatch.setattr("crewai_cli.deploy.main.DeployCommand", FakeDeployCommand)
    monkeypatch.setattr(
        "crewai_cli.authentication.main.AuthenticationCommand",
        FakeAuthenticationCommand,
    )

    run_crew._chain_deploy()

    assert create_calls == [
        {"confirm": False, "skip_validate": True},
        {"confirm": False, "skip_validate": True},
    ]
    assert login_calls == [True]


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


def test_plan_step_lifecycle_events_update_the_explicit_step() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        _emit_event(
            PlanStepStartedEvent(
                agent_role="Agent",
                step_number=2,
                step_description="Second",
            )
        )

        assert app._plan_step_status == {
            1: "pending",
            2: "active",
            3: "pending",
        }

        _emit_event(
            PlanStepCompletedEvent(
                agent_role="Agent",
                step_number=2,
                step_description="Second",
                success=True,
                result="done",
            )
        )
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "pending",
        2: "done",
        3: "pending",
    }


def test_failed_plan_step_lifecycle_event_marks_exact_step_failed() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        _emit_event(
            PlanStepCompletedEvent(
                agent_role="Agent",
                step_number=2,
                step_description="Second",
                success=False,
                error="Step failed",
            )
        )
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "pending",
        2: "failed",
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


def test_next_tool_does_not_mark_unfinished_tool_successful() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        _emit_event(
            ToolUsageStartedEvent(tool_name="search", tool_args={"query": "CrewAI"}),
        )
        _emit_event(
            ToolUsageStartedEvent(tool_name="scrape", tool_args={"url": "https://x"}),
        )
    finally:
        app._unsubscribe()

    assert app._log_entries[0]["status"] == "timeout"
    assert app._log_entries[0]["result"] is None
    assert app._log_entries[0]["error"] == (
        "No result received before the next tool started"
    )
    assert app._log_entries[1]["status"] == "running"
    assert app._plan_step_status == {
        1: "pending",
        2: "pending",
        3: "pending",
    }


def test_internal_reasoning_function_call_is_hidden_from_activity_log() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        future = crewai_event_bus.emit(
            None,
            ToolUsageStartedEvent(
                tool_name="create_reasoning_plan",
                tool_args={"plan": "Plan", "steps": [], "ready": True},
            ),
        )
        if future:
            future.result(timeout=5)

        now = datetime.now()
        future = crewai_event_bus.emit(
            None,
            ToolUsageFinishedEvent(
                tool_name="create_reasoning_plan",
                tool_args={"plan": "Plan", "steps": [], "ready": True},
                started_at=now,
                finished_at=now,
                output='{"plan":"Plan","steps":[],"ready":true}',
            ),
        )
        if future:
            future.result(timeout=5)

        future = crewai_event_bus.emit(
            None,
            ToolUsageErrorEvent(
                tool_name="create_reasoning_plan",
                tool_args={"plan": "Plan", "steps": [], "ready": True},
                error="internal planning fallback",
            ),
        )
        if future:
            future.result(timeout=5)
    finally:
        app._unsubscribe()

    assert app._log_entries == []
    assert app._current_task_steps == []


def test_tool_failure_does_not_override_successful_plan_step_completion() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        _emit_event(
            PlanStepStartedEvent(
                agent_role="Agent",
                step_number=1,
                step_description="First",
            )
        )
        _emit_event(
            ToolUsageStartedEvent(
                tool_name="search_the_internet_with_serper",
                tool_args={"search_query": "CrewAI release"},
                plan_step_number=1,
                plan_step_description="First",
            )
        )
        _emit_event(
            ToolUsageErrorEvent(
                tool_name="search_the_internet_with_serper",
                tool_args={"search_query": "CrewAI release"},
                plan_step_number=1,
                plan_step_description="First",
                error="No results",
            )
        )
        _emit_event(
            PlanStepCompletedEvent(
                agent_role="Agent",
                step_number=1,
                step_description="First",
                success=True,
                result="Recovered with another source",
            )
        )
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "done",
        2: "pending",
        3: "pending",
    }


def test_tool_event_step_metadata_is_stored_in_activity_log() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        _emit_event(
            ToolUsageStartedEvent(
                tool_name="search_the_internet_with_serper",
                tool_args={"search_query": "CrewAI release"},
                plan_step_number=2,
                plan_step_description="Second",
            )
        )
        now = datetime.now()
        _emit_event(
            ToolUsageFinishedEvent(
                tool_name="search_the_internet_with_serper",
                tool_args={"search_query": "CrewAI release"},
                plan_step_number=2,
                plan_step_description="Second",
                started_at=now,
                finished_at=now,
                output="Found official source",
            )
        )
    finally:
        app._unsubscribe()

    assert app._log_entries[0]["plan_step_number"] == 2
    assert app._plan_step_status == {
        1: "pending",
        2: "pending",
        3: "pending",
    }


def test_starting_next_tool_does_not_infer_plan_step_progress() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        _emit_event(
            ToolUsageStartedEvent(
                tool_name="search_the_internet_with_serper",
                tool_args={"search_query": "CrewAI release"},
            )
        )
        _emit_event(
            ToolUsageErrorEvent(
                tool_name="search_the_internet_with_serper",
                tool_args={"search_query": "CrewAI release"},
                error="No results",
            )
        )
        _emit_event(
            ToolUsageStartedEvent(
                tool_name="read_website_content",
                tool_args={"url": "https://example.com"},
            )
        )
    finally:
        app._unsubscribe()

    assert app._log_entries[0]["status"] == "error"
    assert app._log_entries[1]["status"] == "running"
    assert app._plan_step_status == {
        1: "pending",
        2: "pending",
        3: "pending",
    }


@pytest.mark.asyncio
async def test_crew_done_does_not_mark_unfinished_tool_successful() -> None:
    app = _app_with_plan()

    async with app.run_test(size=(100, 40)) as pilot:
        app._plan_step_status = {1: "failed", 2: "done", 3: "pending"}
        app._log_entries = [
            {
                "tool_name": "search",
                "status": "running",
                "args": '{"query": "CrewAI"}',
                "result": None,
                "error": None,
                "start_time": time.time() - 2,
                "duration": None,
                "task_idx": 1,
            }
        ]

        app._on_crew_done("final output")
        await pilot.pause()

    assert app._log_entries[0]["status"] == "timeout"
    assert app._log_entries[0]["result"] is None
    assert app._log_entries[0]["error"] == "No result received before crew completed"
    assert app._plan_step_status == {1: "failed", 2: "done", 3: "done"}


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


def test_streamed_goal_achieved_observation_collapses_remaining_steps_done() -> None:
    app = _app_with_plan()

    updated = app._try_parse_step_observation(
        '{"step_number":2,'
        '"step_completed_successfully":true,'
        '"key_information_learned":"Goal is already satisfied.",'
        '"goal_already_achieved":true}'
    )

    assert updated is True
    assert app._plan_step_status == {
        1: "done",
        2: "done",
        3: "done",
    }


def test_task_completion_collapses_pending_plan_steps_but_preserves_failed() -> None:
    app = _app_with_plan()
    app._plan_step_status = {1: "failed", 2: "done", 3: "pending"}

    app._collapse_plan_on_task_done()

    assert app._plan_step_status == {1: "failed", 2: "done", 3: "done"}


def test_observation_failure_collapses_to_done_because_executor_continues() -> None:
    app = _app_with_plan()
    app._plan_step_status = {1: "done", 2: "active", 3: "pending"}
    app._subscribe()
    try:
        future = crewai_event_bus.emit(
            None,
            StepObservationFailedEvent(
                agent_role="Agent",
                step_number=2,
                step_description="Second",
                error="observer timeout",
            ),
        )
        if future:
            future.result(timeout=5)
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "done",
        2: "done",
        3: "pending",
    }


def test_goal_achieved_event_collapses_remaining_steps_done() -> None:
    app = _app_with_plan()
    app._plan_step_status = {1: "done", 2: "active", 3: "pending"}
    app._subscribe()
    try:
        future = crewai_event_bus.emit(
            None,
            GoalAchievedEarlyEvent(
                agent_role="Agent",
                step_number=2,
                steps_completed=2,
                steps_remaining=1,
            ),
        )
        if future:
            future.result(timeout=5)
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "done",
        2: "done",
        3: "done",
    }


def test_replan_event_keeps_old_plan_until_next_streamed_plan_replaces_it() -> None:
    app = _app_with_plan()
    app._subscribe()
    try:
        future = crewai_event_bus.emit(
            None,
            PlanReplanTriggeredEvent(
                agent_role="Agent",
                step_number=2,
                replan_reason="Need updated sources",
                replan_count=1,
                completed_steps_preserved=1,
            ),
        )
        if future:
            future.result(timeout=5)
    finally:
        app._unsubscribe()

    assert app._plan is not None
    assert app._plan_step_status == {1: "pending", 2: "pending", 3: "pending"}
    assert app._awaiting_replan is True

    app._try_parse_plan(
        '{"plan":"Updated plan","steps":['
        '{"step_number":1,"description":"Updated first"},'
        '{"step_number":2,"description":"Updated second"}]}'
    )

    assert app._plan == {
        "plan": "Updated plan",
        "steps": [
            {"step_number": 1, "description": "Updated first"},
            {"step_number": 2, "description": "Updated second"},
        ],
    }
    assert app._plan_step_status == {1: "pending", 2: "pending"}
    assert app._awaiting_replan is False


def test_plan_refinement_updates_descriptions_without_new_statuses() -> None:
    app = _app_with_plan()
    app._plan_step_status = {1: "done", 2: "active", 3: "pending"}
    app._subscribe()
    try:
        future = crewai_event_bus.emit(
            None,
            PlanRefinementEvent(
                agent_role="Agent",
                step_number=2,
                refined_step_count=1,
                refinements=["Step 3: Write the final answer from verified facts"],
            ),
        )
        if future:
            future.result(timeout=5)
    finally:
        app._unsubscribe()

    assert app._plan_step_status == {
        1: "done",
        2: "done",
        3: "pending",
    }
    assert app._plan["steps"][2]["description"] == (
        "Write the final answer from verified facts"
    )


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


class _FakeTask:
    fingerprint = None

    def __init__(self, task_id: str, name: str) -> None:
        self.id = task_id
        self.name = name
        self.description = name


def test_async_task_completion_marks_the_right_sidebar_row() -> None:
    """Overlapping tasks: completing task 1 while task 2 runs must not
    mark task 2 done, and starting task 2 must not mark task 1 done."""
    from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent
    from crewai.tasks.task_output import TaskOutput

    app = CrewRunApp(total_tasks=2, task_names=["first", "second"])
    app._subscribe()
    try:
        task1 = _FakeTask("id-1", "first")
        task2 = _FakeTask("id-2", "second")

        for task in (task1, task2):
            future = crewai_event_bus.emit(
                None, TaskStartedEvent(context=None, task=task)
            )
            if future:
                future.result(timeout=5)

        # Both started: neither prematurely done
        assert app._task_statuses == {1: "active", 2: "active"}

        future = crewai_event_bus.emit(
            None,
            TaskCompletedEvent(
                output=TaskOutput(description="first", raw="done", agent="a"),
                task=task1,
            ),
        )
        if future:
            future.result(timeout=5)

        assert app._task_statuses == {1: "done", 2: "active"}
    finally:
        app._unsubscribe()


def test_pop_task_state_falls_back_to_current_task() -> None:
    app = CrewRunApp(total_tasks=2, task_names=["first", "second"])
    app._current_task_idx = 2
    app._current_task_desc = "second"

    class _Evt:
        task = None
        task_name = "unknown"

    state = app._pop_task_state(_Evt())
    assert state["idx"] == 2
    assert state["desc"] == "second"


def test_overlapping_task_logs_keep_their_own_state() -> None:
    """Task 1 completing after task 2 started must log its own description,
    agent, and output — and must not steal or reset task 2's stream state."""
    from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent
    from crewai.tasks.task_output import TaskOutput

    app = CrewRunApp(total_tasks=2, task_names=["first", "second"])
    app._subscribe()
    try:
        task1 = _FakeTask("id-1", "first")
        task2 = _FakeTask("id-2", "second")

        for task in (task1, task2):
            future = crewai_event_bus.emit(
                None, TaskStartedEvent(context=None, task=task)
            )
            if future:
                future.result(timeout=5)

        # Task 2 is current and has streamed state in flight
        app._task_full_output = "task two streaming output"
        app._current_task_steps = [{"type": "llm", "summary": "thinking"}]

        future = crewai_event_bus.emit(
            None,
            TaskCompletedEvent(
                output=TaskOutput(
                    description="first", raw="task one result", agent="a1"
                ),
                task=task1,
            ),
        )
        if future:
            future.result(timeout=5)

        # Task 1's entry carries its own identity and output
        entry1 = app._task_logs[-1]
        assert entry1["idx"] == 1
        assert entry1["desc"] == "first"
        assert entry1["output"] == "task one result"
        assert entry1["steps"] == []

        # Task 2's in-flight stream state was not consumed or reset
        assert app._task_full_output == "task two streaming output"
        assert app._current_task_steps == [{"type": "llm", "summary": "thinking"}]

        future = crewai_event_bus.emit(
            None,
            TaskCompletedEvent(
                output=TaskOutput(
                    description="second", raw="task two result", agent="a2"
                ),
                task=task2,
            ),
        )
        if future:
            future.result(timeout=5)

        entry2 = app._task_logs[-1]
        assert entry2["idx"] == 2
        assert entry2["desc"] == "second"
        assert entry2["output"] == "task two streaming output"
        assert any(step.get("summary") == "thinking" for step in entry2["steps"])
    finally:
        app._unsubscribe()
