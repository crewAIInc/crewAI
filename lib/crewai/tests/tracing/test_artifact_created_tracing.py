"""Saved artifacts must reach the trace collector, with who saved them."""

from unittest.mock import patch

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.artifact_events import ArtifactCreatedEvent
import pytest


@pytest.fixture
def handled():
    listener = TraceCollectionListener.__new__(TraceCollectionListener)
    with (
        crewai_event_bus.scoped_handlers(),
        patch.object(TraceCollectionListener, "_handle_action_event") as handled,
    ):
        listener._register_action_event_handlers(crewai_event_bus)
        yield handled


def _event(**kwargs) -> ArtifactCreatedEvent:
    return ArtifactCreatedEvent(
        id="0b9e6c1e-0000-4000-8000-000000000000",
        identifier="q3-report",
        version=1,
        title="Q3 report",
        kind="html",
        content_type="text/html",
        size=12,
        url="file:///artifacts/v1.html",
        **kwargs,
    )


def test_artifact_created_is_forwarded_intact(handled):
    event = _event(task_id="task-1", agent_role="Analyst")

    crewai_event_bus.emit(None, event)
    crewai_event_bus.flush()

    forwarded = [c.args[2] for c in handled.call_args_list if c.args[0] == "artifact_created"]
    assert forwarded == [event]
    assert forwarded[0] is event
