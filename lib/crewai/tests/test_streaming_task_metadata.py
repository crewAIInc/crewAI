"""Tests for streaming chunk task metadata population.

Verifies that _create_stream_chunk uses event-level task_name and task_id
when available, falling back to current_task_info (issue #4347).
"""

from unittest.mock import MagicMock

import pytest

from crewai.events.types.llm_events import LLMCallType, LLMStreamChunkEvent
from crewai.types.streaming import StreamChunkType
from crewai.utilities.streaming import TaskInfo, _create_stream_chunk


def _make_event(
    chunk: str = "hello",
    task_name: str | None = None,
    task_id: str | None = None,
    agent_role: str | None = None,
    agent_id: str | None = None,
) -> LLMStreamChunkEvent:
    """Helper to build an LLMStreamChunkEvent with optional metadata."""
    return LLMStreamChunkEvent(
        chunk=chunk,
        call_id="test-call-id",
        type="llm_stream_chunk",
        task_name=task_name,
        task_id=task_id,
        agent_role=agent_role,
        agent_id=agent_id,
    )


def _make_task_info(
    index: int = 0,
    name: str = "",
    id: str = "",
    agent_role: str = "",
    agent_id: str = "",
) -> TaskInfo:
    """Helper to build a TaskInfo dict."""
    return TaskInfo(
        index=index,
        name=name,
        id=id,
        agent_role=agent_role,
        agent_id=agent_id,
    )


class TestCreateStreamChunkTaskMetadata:
    """Verify task_name and task_id are populated from event when available."""

    def test_event_task_metadata_preferred_over_empty_task_info(self) -> None:
        """When the event carries task metadata, it should appear in the chunk
        even if current_task_info has empty strings (the default)."""
        event = _make_event(
            task_name="Explain Quantum Computing",
            task_id="abc-123",
            agent_role="Scientist",
            agent_id="agent-1",
        )
        task_info = _make_task_info()  # all empty

        chunk = _create_stream_chunk(event, task_info)

        assert chunk.task_name == "Explain Quantum Computing"
        assert chunk.task_id == "abc-123"

    def test_event_task_metadata_preferred_over_task_info(self) -> None:
        """Event-level metadata takes priority over current_task_info."""
        event = _make_event(
            task_name="From Event",
            task_id="event-id",
            agent_role="Role",
            agent_id="aid",
        )
        task_info = _make_task_info(name="From TaskInfo", id="info-id")

        chunk = _create_stream_chunk(event, task_info)

        assert chunk.task_name == "From Event"
        assert chunk.task_id == "event-id"

    def test_fallback_to_task_info_when_event_has_no_metadata(self) -> None:
        """When the event has no task metadata, fall back to current_task_info."""
        event = _make_event(
            task_name=None,
            task_id=None,
        )
        task_info = _make_task_info(
            name="Fallback Task",
            id="fallback-id",
        )

        chunk = _create_stream_chunk(event, task_info)

        assert chunk.task_name == "Fallback Task"
        assert chunk.task_id == "fallback-id"

    def test_agent_metadata_consistency(self) -> None:
        """agent_role and agent_id should continue to prefer event values."""
        event = _make_event(
            agent_role="From Event Role",
            agent_id="event-agent-id",
        )
        task_info = _make_task_info(
            agent_role="Fallback Role",
            agent_id="fallback-agent-id",
        )

        chunk = _create_stream_chunk(event, task_info)

        assert chunk.agent_role == "From Event Role"
        assert chunk.agent_id == "event-agent-id"

    def test_mixed_metadata_sources(self) -> None:
        """task_name from event, task_id from fallback (mixed sources)."""
        event = _make_event(
            task_name="Event Task Name",
            task_id=None,  # not provided
            agent_role="Agent",
            agent_id="aid",
        )
        task_info = _make_task_info(
            name="Info Task Name",
            id="info-task-id",
        )

        chunk = _create_stream_chunk(event, task_info)

        assert chunk.task_name == "Event Task Name"
        assert chunk.task_id == "info-task-id"

    def test_empty_string_event_fields_fallback_to_task_info(self) -> None:
        """Empty-string event fields should fall back to task_info values."""
        event = _make_event(
            task_name="",
            task_id="",
        )
        task_info = _make_task_info(
            name="Fallback Name",
            id="fallback-id",
        )

        chunk = _create_stream_chunk(event, task_info)

        # Empty string is falsy, so should fall back
        assert chunk.task_name == "Fallback Name"
        assert chunk.task_id == "fallback-id"

    def test_chunk_content_and_type_preserved(self) -> None:
        """Core chunk fields should be unaffected by metadata changes."""
        event = _make_event(
            chunk="streaming content",
            task_name="Task",
            task_id="tid",
        )
        task_info = _make_task_info(index=2)

        chunk = _create_stream_chunk(event, task_info)

        assert chunk.content == "streaming content"
        assert chunk.chunk_type == StreamChunkType.TEXT
        assert chunk.task_index == 2
        assert chunk.tool_call is None
