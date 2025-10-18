"""Tests for FastAPI-style dependency injection in event handlers."""

import asyncio

import pytest

from crewai.events import Depends, crewai_event_bus
from crewai.events.base_events import BaseEvent


class DependsTestEvent(BaseEvent):
    """Test event for dependency tests."""

    value: int = 0
    type: str = "test_event"


@pytest.mark.asyncio
async def test_basic_dependency():
    """Test that handler with dependency runs after its dependency."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        def setup(source, event: DependsTestEvent):
            execution_order.append("setup")

        @crewai_event_bus.on(DependsTestEvent, Depends(setup))
        def process(source, event: DependsTestEvent):
            execution_order.append("process")

        event = DependsTestEvent(value=1)
        future = crewai_event_bus.emit("test_source", event)

        if future:
            await asyncio.wrap_future(future)

        assert execution_order == ["setup", "process"]


@pytest.mark.asyncio
async def test_multiple_dependencies():
    """Test handler with multiple dependencies."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        def setup_a(source, event: DependsTestEvent):
            execution_order.append("setup_a")

        @crewai_event_bus.on(DependsTestEvent)
        def setup_b(source, event: DependsTestEvent):
            execution_order.append("setup_b")

        @crewai_event_bus.on(
            DependsTestEvent, depends_on=[Depends(setup_a), Depends(setup_b)]
        )
        def process(source, event: DependsTestEvent):
            execution_order.append("process")

        event = DependsTestEvent(value=1)
        future = crewai_event_bus.emit("test_source", event)

        if future:
            await asyncio.wrap_future(future)

        # setup_a and setup_b can run in any order (same level)
        assert "process" in execution_order
        assert execution_order.index("process") > execution_order.index("setup_a")
        assert execution_order.index("process") > execution_order.index("setup_b")


@pytest.mark.asyncio
async def test_chain_of_dependencies():
    """Test chain of dependencies (A -> B -> C)."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        def handler_a(source, event: DependsTestEvent):
            execution_order.append("handler_a")

        @crewai_event_bus.on(DependsTestEvent, depends_on=Depends(handler_a))
        def handler_b(source, event: DependsTestEvent):
            execution_order.append("handler_b")

        @crewai_event_bus.on(DependsTestEvent, depends_on=Depends(handler_b))
        def handler_c(source, event: DependsTestEvent):
            execution_order.append("handler_c")

        event = DependsTestEvent(value=1)
        future = crewai_event_bus.emit("test_source", event)

        if future:
            await asyncio.wrap_future(future)

        assert execution_order == ["handler_a", "handler_b", "handler_c"]


@pytest.mark.asyncio
async def test_async_handler_with_dependency():
    """Test async handler with dependency on sync handler."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        def sync_setup(source, event: DependsTestEvent):
            execution_order.append("sync_setup")

        @crewai_event_bus.on(DependsTestEvent, depends_on=Depends(sync_setup))
        async def async_process(source, event: DependsTestEvent):
            await asyncio.sleep(0.01)
            execution_order.append("async_process")

        event = DependsTestEvent(value=1)
        future = crewai_event_bus.emit("test_source", event)

        if future:
            await asyncio.wrap_future(future)

        assert execution_order == ["sync_setup", "async_process"]


@pytest.mark.asyncio
async def test_mixed_handlers_with_dependencies():
    """Test mix of sync and async handlers with dependencies."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        def setup(source, event: DependsTestEvent):
            execution_order.append("setup")

        @crewai_event_bus.on(DependsTestEvent, depends_on=Depends(setup))
        def sync_process(source, event: DependsTestEvent):
            execution_order.append("sync_process")

        @crewai_event_bus.on(DependsTestEvent, depends_on=Depends(setup))
        async def async_process(source, event: DependsTestEvent):
            await asyncio.sleep(0.01)
            execution_order.append("async_process")

        @crewai_event_bus.on(
            DependsTestEvent, depends_on=[Depends(sync_process), Depends(async_process)]
        )
        def finalize(source, event: DependsTestEvent):
            execution_order.append("finalize")

        event = DependsTestEvent(value=1)
        future = crewai_event_bus.emit("test_source", event)

        if future:
            await asyncio.wrap_future(future)

        # Verify execution order
        assert execution_order[0] == "setup"
        assert "finalize" in execution_order
        assert execution_order.index("finalize") > execution_order.index("sync_process")
        assert execution_order.index("finalize") > execution_order.index("async_process")


@pytest.mark.asyncio
async def test_independent_handlers_run_concurrently():
    """Test that handlers without dependencies can run concurrently."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        async def handler_a(source, event: DependsTestEvent):
            await asyncio.sleep(0.01)
            execution_order.append("handler_a")

        @crewai_event_bus.on(DependsTestEvent)
        async def handler_b(source, event: DependsTestEvent):
            await asyncio.sleep(0.01)
            execution_order.append("handler_b")

        event = DependsTestEvent(value=1)
        future = crewai_event_bus.emit("test_source", event)

        if future:
            await asyncio.wrap_future(future)

        # Both handlers should have executed
        assert len(execution_order) == 2
        assert "handler_a" in execution_order
        assert "handler_b" in execution_order


@pytest.mark.asyncio
async def test_circular_dependency_detection():
    """Test that circular dependencies are detected and raise an error."""
    from crewai.events.handler_graph import CircularDependencyError, build_execution_plan

    # Create circular dependency: handler_a -> handler_b -> handler_c -> handler_a
    def handler_a(source, event: DependsTestEvent):
        pass

    def handler_b(source, event: DependsTestEvent):
        pass

    def handler_c(source, event: DependsTestEvent):
        pass

    # Build a dependency graph with a cycle
    handlers = [handler_a, handler_b, handler_c]
    dependencies = {
        handler_a: [Depends(handler_b)],
        handler_b: [Depends(handler_c)],
        handler_c: [Depends(handler_a)],  # Creates the cycle
    }

    # Should raise CircularDependencyError about circular dependency
    with pytest.raises(CircularDependencyError, match="Circular dependency"):
        build_execution_plan(handlers, dependencies)


@pytest.mark.asyncio
async def test_handler_without_dependency_runs_normally():
    """Test that handlers without dependencies still work as before."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        def simple_handler(source, event: DependsTestEvent):
            execution_order.append("simple_handler")

        event = DependsTestEvent(value=1)
        future = crewai_event_bus.emit("test_source", event)

        if future:
            await asyncio.wrap_future(future)

        assert execution_order == ["simple_handler"]


@pytest.mark.asyncio
async def test_depends_equality():
    """Test Depends equality and hashing."""

    def handler_a(source, event):
        pass

    def handler_b(source, event):
        pass

    dep_a1 = Depends(handler_a)
    dep_a2 = Depends(handler_a)
    dep_b = Depends(handler_b)

    # Same handler should be equal
    assert dep_a1 == dep_a2
    assert hash(dep_a1) == hash(dep_a2)

    # Different handlers should not be equal
    assert dep_a1 != dep_b
    assert hash(dep_a1) != hash(dep_b)


@pytest.mark.asyncio
async def test_aemit_ignores_dependencies():
    """Test that aemit only processes async handlers (no dependency support yet)."""
    execution_order = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(DependsTestEvent)
        def sync_handler(source, event: DependsTestEvent):
            execution_order.append("sync_handler")

        @crewai_event_bus.on(DependsTestEvent)
        async def async_handler(source, event: DependsTestEvent):
            execution_order.append("async_handler")

        event = DependsTestEvent(value=1)
        await crewai_event_bus.aemit("test_source", event)

        # Only async handler should execute
        assert execution_order == ["async_handler"]
