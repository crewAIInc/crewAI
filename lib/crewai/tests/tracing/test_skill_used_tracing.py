"""Skill usage must reach the trace collector.

The five setup events (discovery, load, activation, failure) describe how an
agent was configured and fire once. ``SkillUsedEvent`` is the only runtime
signal -- it re-fires on every execution -- so without it a trace cannot say
which skills an agent actually used, on which task, or how often.
"""

from pathlib import Path
from unittest.mock import patch

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.skill_events import (
    SkillActivatedEvent,
    SkillUsedEvent,
)
import pytest


@pytest.fixture
def registered_listener():
    """A listener wired to the bus, with event handling captured.

    ``scoped_handlers`` is required, not tidiness: ``CrewAIEventsBus`` is a
    singleton, so registering on a locally constructed one still mutates the
    process-wide bus. Without the scope these handlers outlive the test and
    fire against a listener built with ``__new__`` -- no ``batch_manager`` --
    in whatever runs next.
    """
    listener = TraceCollectionListener.__new__(TraceCollectionListener)

    with (
        crewai_event_bus.scoped_handlers(),
        patch.object(TraceCollectionListener, "_handle_action_event") as handled,
    ):
        listener._register_action_event_handlers(crewai_event_bus)
        yield crewai_event_bus, handled


def _event_types(handled) -> list[str]:
    return [call.args[0] for call in handled.call_args_list]


def _events_of_type(handled, event_type: str) -> list:
    """The event objects forwarded for one collected type."""
    return [call.args[2] for call in handled.call_args_list if call.args[0] == event_type]


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

    def test_the_event_itself_is_forwarded_intact(self, registered_listener):
        """The type alone is not enough -- the collector serializes the event,
        so dropping or replacing it would lose every attribution field.

        Asserted by identity: comparing field values would still pass if a
        handler forwarded a reconstructed copy.
        """
        bus, handled = registered_listener
        event = SkillUsedEvent(
            skill_name="pdf-processing",
            skill_path=Path("/skills/pdf-processing"),
        )

        bus.emit(None, event)
        bus.flush()

        [forwarded] = _events_of_type(handled, "skill_used")
        assert forwarded is event

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
