from unittest.mock import Mock

import pytest

from crewai.utilities.events.base_events import CrewEvent
from crewai.utilities.events.crewai_event_bus import CrewAIEventsBus


class TestEvent(CrewEvent):
    pass

@pytest.fixture
def event_bus():
    bus = CrewAIEventsBus()
    bus.clear_handlers()
    return bus

def test_specific_event_handler(event_bus):
    mock_handler = Mock()

    @event_bus.on(TestEvent)
    def handler(source, event):
        mock_handler(source, event)

    event = TestEvent(type="test_event")
    event_bus.emit("source_object", event)

    mock_handler.assert_called_once_with("source_object", event)

def test_wildcard_event_handler(event_bus):
    mock_handler = Mock()

    @event_bus.on(CrewEvent)
    def handler(source, event):
        mock_handler(source, event)

    event = TestEvent(type="test_event")
    event_bus.emit("source_object", event)

    mock_handler.assert_called_once_with("source_object", event)
