from unittest.mock import Mock

from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus


class TestEvent(BaseEvent):
    pass


def test_specific_event_handler():
    mock_handler = Mock()

    @crewai_event_bus.on(TestEvent)
    def handler(source, event):
        mock_handler(source, event)

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    mock_handler.assert_called_once_with("source_object", event)


def test_wildcard_event_handler():
    mock_handler = Mock()

    @crewai_event_bus.on(BaseEvent)
    def handler(source, event):
        mock_handler(source, event)

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    mock_handler.assert_called_once_with("source_object", event)


def test_event_bus_error_handling(capfd):
    @crewai_event_bus.on(BaseEvent)
    def broken_handler(source, event):
        raise ValueError("Simulated handler failure")

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    out, err = capfd.readouterr()
    assert "Simulated handler failure" in out
    assert "Handler 'broken_handler' failed" in out
