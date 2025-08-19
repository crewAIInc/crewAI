import threading
from typing import Any, Callable, cast
from unittest.mock import Mock

import pytest

from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus


@pytest.fixture(autouse=True)
def scoped_event_handlers():
    with crewai_event_bus.scoped_handlers():
        yield


class TestEvent(BaseEvent):
    pass


class AnotherThreadTestEvent(BaseEvent):
    pass


class CrossThreadTestEvent(BaseEvent):
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
    assert "Global handler 'broken_handler' failed" in out


def test_singleton_pattern_across_threads():
    instances = []

    def get_instance():
        instances.append(crewai_event_bus)

    threads = []
    for _ in range(10):
        thread = threading.Thread(target=get_instance)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    assert len(instances) == 10
    for instance in instances:
        assert instance is crewai_event_bus
        assert instance is instances[0]


def test_default_handlers_are_global():
    """Test that handlers registered with @crewai_event_bus.on() are global by default."""
    received_events = []
    mock_handler = Mock()

    @crewai_event_bus.on(CrossThreadTestEvent)
    def global_handler(source, event):
        received_events.append((source, event))
        mock_handler(source, event)

    def thread_worker(thread_id):
        # Emit event from a different thread
        event = CrossThreadTestEvent(type=f"cross_thread_event_{thread_id}")
        crewai_event_bus.emit(f"thread_source_{thread_id}", event)

    # Start multiple threads that emit events
    threads = []
    for i in range(3):
        thread = threading.Thread(target=thread_worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Verify that the global handler received all events from different threads
    assert len(received_events) == 3
    assert mock_handler.call_count == 3

    # Check that events from different threads were received
    for i in range(3):
        source, event = received_events[i]
        assert source == f"thread_source_{i}"
        assert event.type == f"cross_thread_event_{i}"


def test_scoped_handlers_thread_isolation():
    """Test that scoped_handlers() provides thread-local isolation for testing."""
    global_events = []
    scoped_events = []

    # Register a global handler
    @crewai_event_bus.on(CrossThreadTestEvent)
    def global_handler(source, event):
        global_events.append((source, event))

    # Emit an event - should be received by global handler
    event1 = CrossThreadTestEvent(type="event_1")
    crewai_event_bus.emit("source_1", event1)
    assert len(global_events) == 1

    # Use scoped handlers for testing isolation
    with crewai_event_bus.scoped_handlers():
        # Register a handler in the scoped context (thread-local)
        @crewai_event_bus.on(CrossThreadTestEvent)
        def scoped_handler(source, event):
            scoped_events.append((source, event))

        # Emit event - should be received by scoped handler only
        event2 = CrossThreadTestEvent(type="event_2")
        crewai_event_bus.emit("source_2", event2)

    # After scope, emit another event - should be received by global handler only
    event3 = CrossThreadTestEvent(type="event_3")
    crewai_event_bus.emit("source_3", event3)

    # Verify events
    assert len(global_events) == 2  # event_1 and event_3
    assert len(scoped_events) == 1  # only event_2
    assert global_events[0] == ("source_1", event1)
    assert scoped_events[0] == ("source_2", event2)
    assert global_events[1] == ("source_3", event3)


def test_scoped_handlers_thread_safety():
    """Test that scoped handlers work correctly across multiple threads."""
    thread_results = {}

    def thread_worker(thread_id):
        with crewai_event_bus.scoped_handlers():
            mock_handler = Mock()

            @crewai_event_bus.on(AnotherThreadTestEvent)
            def scoped_handler(source, event):
                mock_handler(f"scoped_thread_{thread_id}", event)

            scoped_event = AnotherThreadTestEvent(type=f"scoped_event_{thread_id}")
            crewai_event_bus.emit(f"scoped_source_{thread_id}", scoped_event)

            thread_results[thread_id] = {
                "mock_handler": mock_handler,
                "scoped_event": scoped_event,
            }

        # After scope, emit event - should not be received by scoped handler
        post_scoped_event = AnotherThreadTestEvent(type=f"post_scoped_{thread_id}")
        crewai_event_bus.emit(f"post_source_{thread_id}", post_scoped_event)

    threads = []
    for i in range(5):
        thread = threading.Thread(target=thread_worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for thread_id, result in thread_results.items():
        result["mock_handler"].assert_called_once_with(
            f"scoped_thread_{thread_id}", result["scoped_event"]
        )


def test_register_handler_method():
    """Test the register_handler method works with global handlers."""
    received_events = []

    def handler(source, event):
        received_events.append((source, event))

    # Register handler using the method
    crewai_event_bus.register_handler(CrossThreadTestEvent, handler)

    # Emit event from different thread
    def thread_worker():
        event = CrossThreadTestEvent(type="test_event")
        crewai_event_bus.emit("thread_source", event)

    thread = threading.Thread(target=thread_worker)
    thread.start()
    thread.join()

    # Verify handler received the event
    assert len(received_events) == 1
    assert received_events[0] == (
        "thread_source",
        CrossThreadTestEvent(type="test_event"),
    )


def test_scoped_global_handlers():
    """Test the scoped_global_handlers context manager."""
    global_events = []

    def global_handler(source, event):
        global_events.append((source, event))

    # Register a global handler
    crewai_event_bus.register_handler(CrossThreadTestEvent, global_handler)

    # Emit an event - should be received
    event1 = CrossThreadTestEvent(type="event_1")
    crewai_event_bus.emit("source_1", event1)
    assert len(global_events) == 1

    # Use scoped global handlers
    with crewai_event_bus.scoped_global_handlers():
        # Register a different handler in scope
        def scoped_handler(source, event):
            global_events.append(("scoped", source, event))

        crewai_event_bus.register_handler(CrossThreadTestEvent, scoped_handler)

        # Emit event - should be received by scoped handler
        event2 = CrossThreadTestEvent(type="event_2")
        crewai_event_bus.emit("source_2", event2)

    # After scope, original handler should be restored
    event3 = CrossThreadTestEvent(type="event_3")
    crewai_event_bus.emit("source_3", event3)

    # Verify events
    assert len(global_events) == 3
    assert global_events[0] == ("source_1", event1)
    assert global_events[1] == ("scoped", "source_2", event2)
    assert global_events[2] == ("source_3", event3)


def test_handler_duplication_scenarios():
    """Test various scenarios where handler duplication can occur."""
    call_counts = []

    def handler(source, event):
        call_counts.append(1)

    # Scenario 1: Register the same handler multiple times
    crewai_event_bus.register_handler(TestEvent, handler)
    crewai_event_bus.register_handler(TestEvent, handler)  # Duplicate registration

    # Scenario 2: Use decorator multiple times on the same function
    @crewai_event_bus.on(TestEvent)
    def decorated_handler1(source, event):
        call_counts.append(1)

    @crewai_event_bus.on(TestEvent)
    def decorated_handler2(source, event):  # Same function name, different instance
        call_counts.append(1)

    # Emit an event
    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source", event)

    # Currently, all handlers are called (including duplicates)
    # This shows the current behavior - handlers can be duplicated
    assert len(call_counts) >= 4  # At least 4 calls (2 direct + 2 decorated)


def test_module_reload_duplication():
    """Test duplication that could occur from module reloading."""
    call_counts = []

    def create_handler():
        def handler(source, event):
            call_counts.append(1)

        return handler

    # Simulate module reload scenario
    handler1 = create_handler()
    handler2 = create_handler()  # Same function, different instance

    crewai_event_bus.register_handler(TestEvent, handler1)
    crewai_event_bus.register_handler(TestEvent, handler2)

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source", event)

    # Both handlers are called (duplication)
    assert len(call_counts) == 2


def test_listener_class_duplication():
    """Test duplication from multiple listener class instances."""
    call_counts = []

    class TestListener:
        def __init__(self):
            @crewai_event_bus.on(TestEvent)
            def handler(source, event):
                call_counts.append(1)

    # Create multiple instances (simulating multiple imports)
    listener1 = TestListener()
    listener2 = TestListener()

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source", event)

    # Both instances register handlers (duplication)
    assert len(call_counts) == 2


def test_handler_deduplication():
    """Test that duplicate handlers are automatically prevented."""
    call_counts = []

    def handler(source, event):
        call_counts.append(1)

    # Register the same handler multiple times
    result1 = crewai_event_bus.register_handler(TestEvent, handler)
    result2 = crewai_event_bus.register_handler(
        TestEvent, handler
    )  # Duplicate registration

    # First registration should succeed, second should fail
    assert result1 is True
    assert result2 is False

    # Emit an event
    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source", event)

    # Handler should only be called once (no duplication)
    assert len(call_counts) == 1


def test_decorator_deduplication():
    """Test that decorator prevents duplicate registrations."""
    call_counts = []

    # Define the same handler function
    def handler(source, event):
        call_counts.append(1)

    # Register using decorator
    @crewai_event_bus.on(TestEvent)
    def decorated_handler(source, event):
        call_counts.append(1)

    # Try to register the same function again using register_handler
    result = crewai_event_bus.register_handler(
        TestEvent, cast(Callable[[Any, BaseEvent], None], decorated_handler)
    )

    # Should fail because it's already registered
    assert result is False

    # Emit an event
    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source", event)

    # Should only be called once
    assert len(call_counts) == 1


def test_handler_unregistration():
    """Test that handlers can be unregistered."""
    call_counts = []

    def handler(source, event):
        call_counts.append(1)

    # Register handler
    crewai_event_bus.register_handler(TestEvent, handler)

    # Verify it's registered
    assert crewai_event_bus.get_handler_count(TestEvent) == 1

    # Emit event - should be called
    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source", event)
    assert len(call_counts) == 1

    # Unregister handler
    result = crewai_event_bus.unregister_handler(TestEvent, handler)
    assert result is True
    assert crewai_event_bus.get_handler_count(TestEvent) == 0

    # Emit event again - should not be called
    crewai_event_bus.emit("source", event)
    assert len(call_counts) == 1  # Still only 1 call


def test_handler_count_tracking():
    """Test that handler counts are tracked correctly."""

    def handler1(source, event):
        pass

    def handler2(source, event):
        pass

    # Initially no handlers
    assert crewai_event_bus.get_handler_count(TestEvent) == 0

    # Register first handler
    crewai_event_bus.register_handler(TestEvent, handler1)
    assert crewai_event_bus.get_handler_count(TestEvent) == 1

    # Register second handler
    crewai_event_bus.register_handler(TestEvent, handler2)
    assert crewai_event_bus.get_handler_count(TestEvent) == 2

    # Try to register first handler again (should fail)
    crewai_event_bus.register_handler(TestEvent, handler1)
    assert crewai_event_bus.get_handler_count(TestEvent) == 2  # Count unchanged

    # Unregister first handler
    crewai_event_bus.unregister_handler(TestEvent, handler1)
    assert crewai_event_bus.get_handler_count(TestEvent) == 1

    # Unregister second handler
    crewai_event_bus.unregister_handler(TestEvent, handler2)
    assert crewai_event_bus.get_handler_count(TestEvent) == 0


def test_different_event_types_dont_conflict():
    """Test that handlers for different event types don't interfere."""
    test_event_calls = []
    cross_thread_calls = []

    def test_event_handler(source, event):
        test_event_calls.append(1)

    def cross_thread_handler(source, event):
        cross_thread_calls.append(1)

    # Register handlers for different event types
    crewai_event_bus.register_handler(TestEvent, test_event_handler)
    crewai_event_bus.register_handler(CrossThreadTestEvent, cross_thread_handler)

    # Emit TestEvent
    test_event = TestEvent(type="test")
    crewai_event_bus.emit("source", test_event)
    assert len(test_event_calls) == 1
    assert len(cross_thread_calls) == 0

    # Emit CrossThreadTestEvent
    cross_thread_event = CrossThreadTestEvent(type="cross_thread")
    crewai_event_bus.emit("source", cross_thread_event)
    assert len(test_event_calls) == 1  # Unchanged
    assert len(cross_thread_calls) == 1


def test_scoped_handlers_with_deduplication():
    """Test that deduplication works within scoped handlers."""
    call_counts = []

    def handler(source, event):
        call_counts.append(1)

    # Register global handler
    crewai_event_bus.register_handler(TestEvent, handler)

    # Use scoped handlers
    with crewai_event_bus.scoped_handlers():
        # Try to register the same handler in scoped context
        @crewai_event_bus.on(TestEvent)
        def scoped_handler(source, event):
            call_counts.append(1)

        # Emit event - should be called by both global and scoped handlers
        event = TestEvent(type="test_event")
        crewai_event_bus.emit("source", event)

    # Should have 2 calls (1 global + 1 scoped)
    assert len(call_counts) == 2
