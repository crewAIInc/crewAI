"""Tests for event bus shutdown and cleanup behavior.

This module tests graceful shutdown, task completion, and cleanup operations.
"""

import asyncio
import threading
import time

import pytest

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus


class ShutdownTestEvent(BaseEvent):
    pass


def test_shutdown_prevents_new_events():
    bus = CrewAIEventsBus()
    received_events = []

    with bus.scoped_handlers():

        @bus.on(ShutdownTestEvent)
        def handler(source: object, event: BaseEvent) -> None:
            received_events.append(event)

        bus._shutting_down = True

        event = ShutdownTestEvent(type="after_shutdown")
        bus.emit("test_source", event)

        time.sleep(0.1)

        assert len(received_events) == 0

        bus._shutting_down = False


@pytest.mark.asyncio
async def test_aemit_during_shutdown():
    bus = CrewAIEventsBus()
    received_events = []

    with bus.scoped_handlers():

        @bus.on(ShutdownTestEvent)
        async def handler(source: object, event: BaseEvent) -> None:
            received_events.append(event)

        bus._shutting_down = True

        event = ShutdownTestEvent(type="aemit_during_shutdown")
        await bus.aemit("test_source", event)

        assert len(received_events) == 0

        bus._shutting_down = False


def test_shutdown_flag_prevents_emit():
    bus = CrewAIEventsBus()
    emitted_count = [0]
    condition = threading.Condition()

    with bus.scoped_handlers():

        @bus.on(ShutdownTestEvent)
        def handler(source: object, event: BaseEvent) -> None:
            with condition:
                emitted_count[0] += 1
                condition.notify()

        event1 = ShutdownTestEvent(type="before_shutdown")
        future = bus.emit("test_source", event1)

        if future:
            future.result(timeout=2.0)

        assert emitted_count[0] == 1

        bus._shutting_down = True

        event2 = ShutdownTestEvent(type="during_shutdown")
        bus.emit("test_source", event2)

        time.sleep(0.1)
        assert emitted_count[0] == 1

        bus._shutting_down = False


def test_concurrent_access_during_shutdown_flag():
    bus = CrewAIEventsBus()
    received_events = []
    condition = threading.Condition()

    with bus.scoped_handlers():

        @bus.on(ShutdownTestEvent)
        def handler(source: object, event: BaseEvent) -> None:
            with condition:
                received_events.append(event)
                condition.notify()

        def emit_events() -> None:
            for i in range(10):
                event = ShutdownTestEvent(type=f"event_{i}")
                bus.emit("source", event)
                time.sleep(0.01)

        def set_shutdown_flag() -> None:
            time.sleep(0.05)
            bus._shutting_down = True

        emit_thread = threading.Thread(target=emit_events)
        shutdown_thread = threading.Thread(target=set_shutdown_flag)

        emit_thread.start()
        shutdown_thread.start()

        emit_thread.join()
        shutdown_thread.join()

        with condition:
            condition.wait_for(lambda: len(received_events) > 0, timeout=2)

        assert len(received_events) < 10
        assert len(received_events) > 0

        bus._shutting_down = False


@pytest.mark.asyncio
async def test_async_handlers_complete_before_shutdown_flag():
    bus = CrewAIEventsBus()
    completed_handlers = []

    with bus.scoped_handlers():

        @bus.on(ShutdownTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.05)
            if not bus._shutting_down:
                completed_handlers.append(event)

        for i in range(5):
            event = ShutdownTestEvent(type=f"event_{i}")
            bus.emit("source", event)

        await asyncio.sleep(0.3)

        assert len(completed_handlers) == 5


def test_scoped_handlers_cleanup():
    bus = CrewAIEventsBus()
    received_before = []
    received_during = []
    received_after = []
    condition = threading.Condition()

    with bus.scoped_handlers():

        @bus.on(ShutdownTestEvent)
        def before_handler(source: object, event: BaseEvent) -> None:
            with condition:
                received_before.append(event)
                condition.notify()

        with bus.scoped_handlers():

            @bus.on(ShutdownTestEvent)
            def during_handler(source: object, event: BaseEvent) -> None:
                with condition:
                    received_during.append(event)
                    condition.notify()

            event1 = ShutdownTestEvent(type="during")
            bus.emit("source", event1)

            with condition:
                condition.wait_for(lambda: len(received_during) >= 1, timeout=2)

            assert len(received_before) == 0
            assert len(received_during) == 1

        event2 = ShutdownTestEvent(type="after_inner_scope")
        bus.emit("source", event2)

        with condition:
            condition.wait_for(lambda: len(received_before) >= 1, timeout=2)

        assert len(received_before) == 1
        assert len(received_during) == 1

    event3 = ShutdownTestEvent(type="after_outer_scope")
    bus.emit("source", event3)

    with condition:
        condition.wait(timeout=0.2)

    assert len(received_before) == 1
    assert len(received_during) == 1
    assert len(received_after) == 0


def test_handler_registration_thread_safety():
    bus = CrewAIEventsBus()
    handlers_registered = [0]
    lock = threading.Lock()

    with bus.scoped_handlers():

        def register_handlers() -> None:
            for _ in range(20):

                @bus.on(ShutdownTestEvent)
                def handler(source: object, event: BaseEvent) -> None:
                    pass

                with lock:
                    handlers_registered[0] += 1

                time.sleep(0.001)

        threads = [threading.Thread(target=register_handlers) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert handlers_registered[0] == 60


@pytest.mark.asyncio
async def test_mixed_sync_async_handler_execution():
    bus = CrewAIEventsBus()
    sync_executed = []
    async_executed = []
    condition = threading.Condition()

    with bus.scoped_handlers():

        @bus.on(ShutdownTestEvent)
        def sync_handler(source: object, event: BaseEvent) -> None:
            time.sleep(0.01)
            with condition:
                sync_executed.append(event)
                condition.notify()

        @bus.on(ShutdownTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.01)
            with condition:
                async_executed.append(event)
                condition.notify()

        for i in range(5):
            event = ShutdownTestEvent(type=f"event_{i}")
            bus.emit("source", event)

        def wait_for_completion():
            with condition:
                return condition.wait_for(
                    lambda: len(sync_executed) == 5 and len(async_executed) == 5,
                    timeout=5
                )

        await asyncio.get_event_loop().run_in_executor(None, wait_for_completion)

        assert len(sync_executed) == 5
        assert len(async_executed) == 5
