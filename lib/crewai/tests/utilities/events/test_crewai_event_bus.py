import socket
import threading
from unittest.mock import Mock

import pytest

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus, crewai_event_bus


class TestEvent(BaseEvent):
    pass


def test_executor_init_survives_socket_blocking_guards(monkeypatch):
    """Event-bus startup must not be broken by socket-blocking test guards.

    Windows emulates ``socket.socketpair()`` with a real ``connect()`` to
    127.0.0.1, and asyncio builds every event loop's self-pipe that way.
    Socket-blocking guards (pytest-recording's ``--block-network``,
    pytest-socket's ``--disable-socket``) replace ``socket.socket.connect``
    and reject that internal, non-network self-pipe, so the first ``emit()``
    dies with "Network is disabled" before any test logic runs. Linux builds
    the self-pipe via ``os.socketpair()`` and never calls ``connect()``,
    which is why CI never sees this. The event bus must initialize under the
    guard and put the guard back afterwards.
    """

    def _network_guard(self, address, *args, **kwargs):
        raise RuntimeError("Network is disabled")

    monkeypatch.setattr(socket.socket, "connect", _network_guard)
    monkeypatch.setattr(socket.socket, "connect_ex", _network_guard)

    bus = CrewAIEventsBus()
    bus._ensure_executor_initialized()

    assert bus._loop is not None
    bus._loop.call_soon_threadsafe(lambda: None)

    # The guard must be reinstated once the loop exists.
    with pytest.raises(RuntimeError, match="Network is disabled"):
        socket.socket().connect(("127.0.0.1", 1))

    monkeypatch.undo()


def test_specific_event_handler():
    mock_handler = Mock()
    condition = threading.Condition()
    handler_called = [False]

    @crewai_event_bus.on(TestEvent)
    def handler(source, event):
        with condition:
            mock_handler(source, event)
            handler_called[0] = True
            condition.notify()

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    with condition:
        if not handler_called[0]:
            condition.wait(timeout=5)

    assert handler_called[0], "Handler was not called within timeout"
    mock_handler.assert_called_once_with("source_object", event)


def test_multiple_handlers_same_event():
    """Test that multiple handlers can be registered for the same event type."""
    mock_handler1 = Mock()
    mock_handler2 = Mock()
    condition = threading.Condition()
    handlers_called = {"handler1": False, "handler2": False}

    @crewai_event_bus.on(TestEvent)
    def handler1(source, event):
        with condition:
            mock_handler1(source, event)
            handlers_called["handler1"] = True
            condition.notify()

    @crewai_event_bus.on(TestEvent)
    def handler2(source, event):
        with condition:
            mock_handler2(source, event)
            handlers_called["handler2"] = True
            condition.notify()

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    with condition:
        while not all(handlers_called.values()):
            condition.wait(timeout=5)
            if not all(handlers_called.values()):
                break

    assert handlers_called["handler1"], "Handler1 was not called within timeout"
    assert handlers_called["handler2"], "Handler2 was not called within timeout"
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
