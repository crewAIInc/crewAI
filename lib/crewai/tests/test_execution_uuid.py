"""Tests for OSS execution uuid creation and nesting inheritance."""

from __future__ import annotations

import pytest

from crewai.execution import (
    _current_execution_uuid,
    begin_execution,
    clear_execution_uuid,
    end_execution,
    get_execution_uuid,
    set_execution_uuid,
)


@pytest.fixture(autouse=True)
def _isolate_execution_uuid() -> None:
    token = _current_execution_uuid.set(None)
    yield
    _current_execution_uuid.reset(token)


def test_begin_creates_when_empty() -> None:
    assert get_execution_uuid() is None
    first_token = begin_execution()
    first = get_execution_uuid()
    second_token = begin_execution()
    assert first
    assert get_execution_uuid() == first
    assert second_token is None
    end_execution(second_token)
    end_execution(first_token)


def test_begin_does_not_overwrite_existing() -> None:
    token = set_execution_uuid("enterprise-kickoff-id")
    try:
        assert begin_execution("should-not-win") is None
        assert get_execution_uuid() == "enterprise-kickoff-id"
    finally:
        clear_execution_uuid(token)


def test_set_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        set_execution_uuid("")


def test_nested_execution_inherits_and_only_owner_clears() -> None:
    parent_token = begin_execution()
    parent_id = get_execution_uuid()
    child_token = begin_execution()

    assert parent_id
    assert child_token is None
    assert get_execution_uuid() == parent_id

    end_execution(child_token)
    assert get_execution_uuid() == parent_id

    end_execution(parent_token)
    assert get_execution_uuid() is None


def test_two_sequential_outer_runs_get_distinct_uuids() -> None:
    first_token = begin_execution()
    first_id = get_execution_uuid()
    end_execution(first_token)
    second_token = begin_execution()
    second_id = get_execution_uuid()
    end_execution(second_token)

    assert first_id != second_id


def test_flow_kickoff_creates_and_clears_execution_uuid() -> None:
    from crewai.flow.flow import Flow, start

    seen: dict[str, str | None] = {}

    class ProbeFlow(Flow):
        @start()
        def begin(self) -> str:
            seen["during"] = get_execution_uuid()
            return "ok"

    flow = ProbeFlow()
    assert get_execution_uuid() is None
    flow.kickoff()
    assert seen["during"]
    assert get_execution_uuid() is None


def test_flow_kickoff_inherits_enterprise_execution_uuid() -> None:
    from crewai.flow.flow import Flow, start

    seen: dict[str, str | None] = {}

    class ProbeFlow(Flow):
        @start()
        def begin(self) -> str:
            seen["during"] = get_execution_uuid()
            return "ok"

    token = set_execution_uuid("celery-kickoff-id")
    try:
        ProbeFlow().kickoff()
        assert seen["during"] == "celery-kickoff-id"
        # Owner was enterprise set(), not the flow — still set here.
        assert get_execution_uuid() == "celery-kickoff-id"
    finally:
        clear_execution_uuid(token)

    assert get_execution_uuid() is None


@pytest.mark.asyncio
async def test_crew_akickoff_creates_and_clears_execution_uuid() -> None:
    from unittest.mock import patch

    from crewai import Agent, Crew, Task
    from crewai.tasks.task_output import TaskOutput

    seen: dict[str, str | None] = {}
    agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    async def capture(*_args: object, **_kwargs: object) -> TaskOutput:
        seen["during"] = get_execution_uuid()
        return TaskOutput(description="d", raw="ok", agent="r")

    with patch("crewai.task.Task.aexecute_sync", side_effect=capture):
        assert get_execution_uuid() is None
        await crew.akickoff()

    assert seen["during"]
    assert get_execution_uuid() is None


@pytest.mark.asyncio
async def test_crew_akickoff_inherits_enterprise_execution_uuid() -> None:
    from unittest.mock import patch

    from crewai import Agent, Crew, Task
    from crewai.tasks.task_output import TaskOutput

    seen: dict[str, str | None] = {}
    agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    async def capture(*_args: object, **_kwargs: object) -> TaskOutput:
        seen["during"] = get_execution_uuid()
        return TaskOutput(description="d", raw="ok", agent="r")

    token = set_execution_uuid("celery-kickoff-id")
    try:
        with patch("crewai.task.Task.aexecute_sync", side_effect=capture):
            await crew.akickoff()
        assert seen["during"] == "celery-kickoff-id"
        assert get_execution_uuid() == "celery-kickoff-id"
    finally:
        clear_execution_uuid(token)

    assert get_execution_uuid() is None


def test_flow_pause_persists_execution_uuid_and_resume_restores_it() -> None:
    import os
    import tempfile

    from crewai.flow import Flow, human_feedback, listen, start
    from crewai.flow.async_feedback.types import (
        HumanFeedbackPending,
        PendingFeedbackContext,
    )
    from crewai.flow.persistence import SQLiteFlowPersistence

    seen: dict[str, str | None] = {}

    class PausingProvider:
        def request_feedback(
            self, context: PendingFeedbackContext, flow: Flow
        ) -> str:
            raise HumanFeedbackPending(context=context)

    class ReviewFlow(Flow):
        @start()
        @human_feedback(message="Review:", provider=PausingProvider())
        def generate(self) -> str:
            seen["during_kickoff"] = get_execution_uuid()
            return "draft"

        @listen(generate)
        def process(self, result: object) -> str:
            seen["during_resume"] = get_execution_uuid()
            return "done"

    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = SQLiteFlowPersistence(os.path.join(tmpdir, "test.db"))
        pending = ReviewFlow(persistence=persistence).kickoff()
        assert isinstance(pending, HumanFeedbackPending)
        paused_id = seen["during_kickoff"]
        assert paused_id
        assert pending.context.execution_uuid == paused_id
        assert get_execution_uuid() is None

        ReviewFlow.from_pending(pending.context.flow_id, persistence).resume("ok")
        assert seen["during_resume"] == paused_id
        assert get_execution_uuid() is None


def test_pending_feedback_context_roundtrips_execution_uuid() -> None:
    from crewai.flow.async_feedback.types import PendingFeedbackContext

    original = PendingFeedbackContext(
        flow_id="flow-1",
        flow_class="test.Flow",
        method_name="review",
        method_output="draft",
        message="Review:",
        execution_uuid="kickoff-uuid",
    )
    restored = PendingFeedbackContext.from_dict(original.to_dict())
    assert restored.execution_uuid == "kickoff-uuid"

    legacy = PendingFeedbackContext.from_dict(
        {
            "flow_id": "flow-1",
            "flow_class": "test.Flow",
            "method_name": "review",
            "method_output": "draft",
            "message": "Review:",
        }
    )
    assert legacy.execution_uuid is None
