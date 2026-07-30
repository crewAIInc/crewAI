"""Skill usage must reach the trace collector.

The five setup events (discovery, load, activation, failure) describe how an
agent was configured and fire once. ``SkillUsedEvent`` is the only runtime
signal -- it re-fires on every execution -- so without it a trace cannot say
which skills an agent actually used, on which task, or how often.
"""

from pathlib import Path
from unittest.mock import patch

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.skill_events import (
    SkillActivatedEvent,
    SkillUsedEvent,
)
import pytest


@pytest.fixture
def registered_listener():
    """A listener wired to an isolated bus, with event handling captured."""
    bus = CrewAIEventsBus()
    listener = TraceCollectionListener.__new__(TraceCollectionListener)

    with patch.object(TraceCollectionListener, "_handle_action_event") as handled:
        listener._register_action_event_handlers(bus)
        yield bus, handled


def _event_types(handled) -> list[str]:
    return [call.args[0] for call in handled.call_args_list]


class TestSkillUsedIsCollected:
    def test_skill_used_reaches_the_collector(self, registered_listener):
        bus, handled = registered_listener

        bus.emit(
            None,
            SkillUsedEvent(
                skill_name="pdf-processing",
                skill_path=Path("/skills/pdf-processing"),
            ),
        )
        bus.flush()

        assert "skill_used" in _event_types(handled), (
            "SkillUsedEvent was emitted but the trace listener ignored it"
        )

    def test_every_use_is_collected(self, registered_listener):
        """Activation is idempotent; usage is not. One event per use."""
        bus, handled = registered_listener

        for _ in range(3):
            bus.emit(None, SkillUsedEvent(skill_name="pdf-processing"))
        bus.flush()

        assert _event_types(handled).count("skill_used") == 3

    def test_setup_events_are_still_collected(self, registered_listener):
        bus, handled = registered_listener

        bus.emit(None, SkillActivatedEvent(skill_name="pdf-processing"))
        bus.flush()

        assert "skill_activated" in _event_types(handled)
