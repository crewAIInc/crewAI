"""Tests for thread safety in CrewAI event bus.

This module tests concurrent event emission and handler registration.
"""

import threading
import time
from collections.abc import Callable

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class ThreadSafetyTestEvent(BaseEvent):
    pass


def test_concurrent_emit_from_multiple_threads():
    received_events: list[BaseEvent] = []
    lock = threading.Lock()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ThreadSafetyTestEvent)
        def handler(source: object, event: BaseEvent) -> None:
            with lock:
                received_events.append(event)

        threads: list[threading.Thread] = []
        num_threads = 10
        events_per_thread = 10

        def emit_events(thread_id: int) -> None:
            for i in range(events_per_thread):
                event = ThreadSafetyTestEvent(type=f"thread_{thread_id}_event_{i}")
                crewai_event_bus.emit(f"source_{thread_id}", event)

        for i in range(num_threads):
            thread = threading.Thread(target=emit_events, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        time.sleep(0.5)

        assert len(received_events) == num_threads * events_per_thread


def test_concurrent_handler_registration():
    handlers_executed: list[int] = []
    lock = threading.Lock()

    def create_handler(handler_id: int) -> Callable[[object, BaseEvent], None]:
        def handler(source: object, event: BaseEvent) -> None:
            with lock:
                handlers_executed.append(handler_id)

        return handler

    with crewai_event_bus.scoped_handlers():
        threads: list[threading.Thread] = []
        num_handlers = 20

        def register_handler(handler_id: int) -> None:
            crewai_event_bus.register_handler(
                ThreadSafetyTestEvent, create_handler(handler_id)
            )

        for i in range(num_handlers):
            thread = threading.Thread(target=register_handler, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        event = ThreadSafetyTestEvent(type="registration_test")
        crewai_event_bus.emit("test_source", event)

        time.sleep(0.5)

        assert len(handlers_executed) == num_handlers
        assert set(handlers_executed) == set(range(num_handlers))


def test_concurrent_emit_and_registration():
    received_events: list[BaseEvent] = []
    lock = threading.Lock()

    with crewai_event_bus.scoped_handlers():

        def emit_continuously() -> None:
            for i in range(50):
                event = ThreadSafetyTestEvent(type=f"emit_event_{i}")
                crewai_event_bus.emit("emitter", event)
                time.sleep(0.001)

        def register_continuously() -> None:
            for _ in range(10):

                @crewai_event_bus.on(ThreadSafetyTestEvent)
                def handler(source: object, event: BaseEvent) -> None:
                    with lock:
                        received_events.append(event)

                time.sleep(0.005)

        emit_thread = threading.Thread(target=emit_continuously)
        register_thread = threading.Thread(target=register_continuously)

        emit_thread.start()
        register_thread.start()

        emit_thread.join()
        register_thread.join()

        time.sleep(0.5)

        assert len(received_events) > 0


def test_stress_test_rapid_emit():
    received_count = [0]
    lock = threading.Lock()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ThreadSafetyTestEvent)
        def counter_handler(source: object, event: BaseEvent) -> None:
            with lock:
                received_count[0] += 1

        num_events = 1000

        for i in range(num_events):
            event = ThreadSafetyTestEvent(type=f"rapid_event_{i}")
            crewai_event_bus.emit("rapid_source", event)

        time.sleep(1.0)

        assert received_count[0] == num_events


def test_multiple_event_types_concurrent():
    class EventTypeA(BaseEvent):
        pass

    class EventTypeB(BaseEvent):
        pass

    received_a: list[BaseEvent] = []
    received_b: list[BaseEvent] = []
    lock = threading.Lock()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(EventTypeA)
        def handler_a(source: object, event: BaseEvent) -> None:
            with lock:
                received_a.append(event)

        @crewai_event_bus.on(EventTypeB)
        def handler_b(source: object, event: BaseEvent) -> None:
            with lock:
                received_b.append(event)

        def emit_type_a() -> None:
            for i in range(50):
                crewai_event_bus.emit("source_a", EventTypeA(type=f"type_a_{i}"))

        def emit_type_b() -> None:
            for i in range(50):
                crewai_event_bus.emit("source_b", EventTypeB(type=f"type_b_{i}"))

        thread_a = threading.Thread(target=emit_type_a)
        thread_b = threading.Thread(target=emit_type_b)

        thread_a.start()
        thread_b.start()

        thread_a.join()
        thread_b.join()

        time.sleep(0.5)

        assert len(received_a) == 50
        assert len(received_b) == 50
