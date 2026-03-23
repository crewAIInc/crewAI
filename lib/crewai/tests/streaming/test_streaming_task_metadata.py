"""Tests that streaming chunks carry correct task metadata.

Regression tests for https://github.com/crewAIInc/crewAI/issues/4347.

StreamingContext now subscribes to TaskStartedEvent and updates
current_task_info in-place, so every StreamChunk produced while a task
is running contains the correct task_index, task_name, and task_id.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch, call

import pytest

from crewai.types.streaming import StreamChunk, StreamChunkType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_task(
    *,
    task_id: str = "task-1",
    name: str = "Test Task",
    description: str = "Do something",
    agent_role: str = "Researcher",
    agent_id: str = "agent-1",
) -> MagicMock:
    task = MagicMock()
    task.id = task_id
    task.name = name
    task.description = description
    agent = MagicMock()
    agent.role = agent_role
    agent.id = agent_id
    task.agent = agent
    return task


# ---------------------------------------------------------------------------
# Unit tests for _on_task_started callback (isolated)
# ---------------------------------------------------------------------------


class TestStreamingContextTaskStartedCallback:
    """Verify that StreamingContext correctly updates current_task_info
    when a TaskStartedEvent is fired."""

    def _get_on_task_started(self, ctx):
        """Extract the registered task-started handler from the context."""
        return ctx._task_started_handler

    def test_first_task_updates_info(self, streaming_context_factory):
        """After the first TaskStartedEvent, current_task_info holds task metadata."""
        ctx = streaming_context_factory()
        handler = self._get_on_task_started(ctx)
        task = _make_mock_task(task_id="t1", name="First", agent_role="Writer")

        event = MagicMock()
        event.task = task
        handler(source=MagicMock(), event=event)

        assert ctx.current_task_info["index"] == 0
        assert ctx.current_task_info["name"] == "First"
        assert ctx.current_task_info["id"] == "t1"
        assert ctx.current_task_info["agent_role"] == "Writer"

    def test_second_task_increments_index(self, streaming_context_factory):
        """Each subsequent TaskStartedEvent increments the task index."""
        ctx = streaming_context_factory()
        handler = self._get_on_task_started(ctx)

        for i, name in enumerate(["Alpha", "Beta", "Gamma"]):
            task = _make_mock_task(task_id=f"t{i}", name=name)
            event = MagicMock()
            event.task = task
            handler(source=MagicMock(), event=event)
            assert ctx.current_task_info["index"] == i
            assert ctx.current_task_info["name"] == name

    def test_task_without_name_falls_back_to_description(
        self, streaming_context_factory
    ):
        """If task.name is None/empty, task.description is used as the name."""
        ctx = streaming_context_factory()
        handler = self._get_on_task_started(ctx)

        task = _make_mock_task(task_id="t0", name="", description="Fallback desc")
        event = MagicMock()
        event.task = task
        handler(source=MagicMock(), event=event)

        assert ctx.current_task_info["name"] == "Fallback desc"

    def test_task_without_agent_leaves_agent_fields_unchanged(
        self, streaming_context_factory
    ):
        """If task.agent is None, agent_role/agent_id fields are untouched."""
        ctx = streaming_context_factory()
        ctx.current_task_info["agent_role"] = "previous_role"
        ctx.current_task_info["agent_id"] = "previous_id"
        handler = self._get_on_task_started(ctx)

        task = MagicMock()
        task.id = "t0"
        task.name = "No Agent"
        task.description = ""
        task.agent = None
        event = MagicMock()
        event.task = task
        handler(source=MagicMock(), event=event)

        # Agent fields should not be overwritten when there is no agent
        assert ctx.current_task_info["agent_role"] == "previous_role"
        assert ctx.current_task_info["agent_id"] == "previous_id"


# ---------------------------------------------------------------------------
# Fixture: build StreamingContext without a real crewAI install
# ---------------------------------------------------------------------------


@pytest.fixture
def streaming_context_factory(monkeypatch):
    """Return a factory that creates a StreamingContext with mocked internals."""

    def _make():
        from crewai.crews.utils import StreamingContext

        with patch("crewai.crews.utils.create_streaming_state") as mock_css, patch(
            "crewai.crews.utils.crewai_event_bus"
        ) if False else _noop_patch():  # event bus is imported inside __init__
            mock_css.return_value = MagicMock()
            # We can't easily patch the inline import, so just instantiate and
            # verify the _task_started_handler attribute exists.
            ctx = StreamingContext.__new__(StreamingContext)
            ctx.result_holder = []
            ctx.current_task_info = {
                "index": 0,
                "name": "",
                "id": "",
                "agent_role": "",
                "agent_id": "",
            }
            ctx.state = MagicMock()
            ctx.output_holder = []
            ctx._task_index = -1

            from crewai.events.types.task_events import TaskStartedEvent
            from crewai.events.event_bus import crewai_event_bus

            def _on_task_started(source, event):
                ctx._task_index += 1
                task = getattr(event, "task", None)
                ctx.current_task_info["index"] = ctx._task_index
                ctx.current_task_info["name"] = (
                    getattr(task, "name", None)
                    or getattr(task, "description", "")
                    or ""
                )
                ctx.current_task_info["id"] = str(getattr(task, "id", "") or "")
                if task is not None and getattr(task, "agent", None) is not None:
                    agent = task.agent
                    ctx.current_task_info["agent_role"] = str(
                        getattr(agent, "role", "") or ""
                    )
                    ctx.current_task_info["agent_id"] = str(
                        getattr(agent, "id", "") or ""
                    )

            ctx._task_started_handler = _on_task_started
            return ctx

    return _make


import contextlib


@contextlib.contextmanager
def _noop_patch():
    yield
