import threading
from unittest.mock import Mock

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


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


def test_multiple_handlers_same_event():
    """Test that multiple handlers can be registered for the same event type."""
    mock_handler1 = Mock()
    mock_handler2 = Mock()

    @crewai_event_bus.on(TestEvent)
    def handler1(source, event):
        mock_handler1(source, event)

    @crewai_event_bus.on(TestEvent)
    def handler2(source, event):
        mock_handler2(source, event)

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    mock_handler1.assert_called_once_with("source_object", event)
    mock_handler2.assert_called_once_with("source_object", event)


def test_event_bus_error_handling():
    """Test that handler exceptions are caught and don't break the event bus."""
    called = threading.Event()
    error_caught = threading.Event()

    @crewai_event_bus.on(TestEvent)
    def broken_handler(source, event):
        called.set()
        raise ValueError("Simulated handler failure")

    @crewai_event_bus.on(TestEvent)
    def working_handler(source, event):
        error_caught.set()

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    assert called.wait(timeout=2), "Broken handler was never called"
    assert error_caught.wait(timeout=2), "Working handler was never called after error"
