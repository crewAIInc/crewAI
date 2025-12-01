"""Tests for async event handling in CrewAI event bus.

This module tests async handler registration, execution, and the aemit method.
"""

import asyncio

import pytest

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class AsyncTestEvent(BaseEvent):
    pass


@pytest.mark.asyncio
async def test_async_handler_execution():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AsyncTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.01)
            received_events.append(event)

        event = AsyncTestEvent(type="async_test")
        crewai_event_bus.emit("test_source", event)

        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0] == event


@pytest.mark.asyncio
async def test_aemit_with_async_handlers():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AsyncTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.01)
            received_events.append(event)

        event = AsyncTestEvent(type="async_test")
        await crewai_event_bus.aemit("test_source", event)

        assert len(received_events) == 1
        assert received_events[0] == event


@pytest.mark.asyncio
async def test_multiple_async_handlers():
    received_events_1 = []
    received_events_2 = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AsyncTestEvent)
        async def handler_1(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.01)
            received_events_1.append(event)

        @crewai_event_bus.on(AsyncTestEvent)
        async def handler_2(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.02)
            received_events_2.append(event)

        event = AsyncTestEvent(type="async_test")
        await crewai_event_bus.aemit("test_source", event)

        assert len(received_events_1) == 1
        assert len(received_events_2) == 1


@pytest.mark.asyncio
async def test_mixed_sync_and_async_handlers():
    sync_events = []
    async_events = []
    sync_done = asyncio.Event()
    async_done = asyncio.Event()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AsyncTestEvent)
        def sync_handler(source: object, event: BaseEvent) -> None:
            sync_events.append(event)
            sync_done.set()

        @crewai_event_bus.on(AsyncTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.01)
            async_events.append(event)
            async_done.set()

        event = AsyncTestEvent(type="mixed_test")
        crewai_event_bus.emit("test_source", event)

        await asyncio.wait_for(sync_done.wait(), timeout=5)
        await asyncio.wait_for(async_done.wait(), timeout=5)

        assert len(sync_events) == 1
        assert len(async_events) == 1


@pytest.mark.asyncio
async def test_async_handler_error_handling():
    successful_handler_called = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AsyncTestEvent)
        async def failing_handler(source: object, event: BaseEvent) -> None:
            raise ValueError("Async handler error")

        @crewai_event_bus.on(AsyncTestEvent)
        async def successful_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.01)
            successful_handler_called.append(True)

        event = AsyncTestEvent(type="error_test")
        await crewai_event_bus.aemit("test_source", event)

        assert len(successful_handler_called) == 1


@pytest.mark.asyncio
async def test_aemit_with_no_handlers():
    with crewai_event_bus.scoped_handlers():
        event = AsyncTestEvent(type="no_handlers")
        await crewai_event_bus.aemit("test_source", event)


@pytest.mark.asyncio
async def test_async_handler_registration_via_register_handler():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        async def custom_async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.01)
            received_events.append(event)

        crewai_event_bus.register_handler(AsyncTestEvent, custom_async_handler)

        event = AsyncTestEvent(type="register_test")
        await crewai_event_bus.aemit("test_source", event)

        assert len(received_events) == 1
        assert received_events[0] == event


@pytest.mark.asyncio
async def test_emit_async_handlers_fire_and_forget():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AsyncTestEvent)
        async def slow_async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.sleep(0.05)
            received_events.append(event)

        event = AsyncTestEvent(type="fire_forget_test")
        crewai_event_bus.emit("test_source", event)

        assert len(received_events) == 0

        await asyncio.sleep(0.1)

        assert len(received_events) == 1


@pytest.mark.asyncio
async def test_scoped_handlers_with_async():
    received_before = []
    received_during = []
    received_after = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(AsyncTestEvent)
        async def before_handler(source: object, event: BaseEvent) -> None:
            received_before.append(event)

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(AsyncTestEvent)
            async def scoped_handler(source: object, event: BaseEvent) -> None:
                received_during.append(event)

            event1 = AsyncTestEvent(type="during_scope")
            await crewai_event_bus.aemit("test_source", event1)

            assert len(received_before) == 0
            assert len(received_during) == 1

        @crewai_event_bus.on(AsyncTestEvent)
        async def after_handler(source: object, event: BaseEvent) -> None:
            received_after.append(event)

        event2 = AsyncTestEvent(type="after_scope")
        await crewai_event_bus.aemit("test_source", event2)

        assert len(received_before) == 1
        assert len(received_during) == 1
        assert len(received_after) == 1
