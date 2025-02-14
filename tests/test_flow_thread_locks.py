"""Tests for Flow with thread locks."""
import asyncio
import threading
from typing import Optional
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, field_validator

from crewai.flow.flow import Flow, start, listen


class ThreadSafeState(BaseModel):
    """Test state model with thread locks."""
    model_config = {
        "arbitrary_types_allowed": True,
        "exclude": {"lock"}
    }
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    lock: Optional[threading.RLock] = Field(default=None, exclude=True)
    value: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.lock is None:
            self.lock = threading.RLock()


class LockFlow(Flow[ThreadSafeState]):
    """Test flow with thread locks."""
    initial_state = ThreadSafeState

    @start()
    async def step_1(self):
        with self.state.lock:
            self.state.value = "step 1"
            return "step 1"

    @listen(step_1)
    async def step_2(self, result):
        with self.state.lock:
            self.state.value += " -> step 2"
            return result + " -> step 2"


def test_flow_with_thread_locks():
    """Test Flow with thread locks in state."""
    flow = LockFlow()
    result = asyncio.run(flow.kickoff_async())
    assert result == "step 1 -> step 2"
    assert flow.state.value == "step 1 -> step 2"


def test_kickoff_async_with_lock_inputs():
    """Test kickoff_async with thread lock inputs."""
    flow = LockFlow()
    inputs = {
        "lock": threading.RLock(),
        "value": "test"
    }
    result = asyncio.run(flow.kickoff_async(inputs=inputs))
    assert result == "step 1 -> step 2"
    assert flow.state.value == "step 1 -> step 2"


class ComplexState(BaseModel):
    """Test state model with nested thread locks."""
    model_config = {
        "arbitrary_types_allowed": True,
        "exclude": {"outer_lock"}
    }
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    outer_lock: Optional[threading.RLock] = Field(default=None, exclude=True)
    inner: Optional[ThreadSafeState] = Field(default_factory=ThreadSafeState)
    value: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.outer_lock is None:
            self.outer_lock = threading.RLock()


class NestedLockFlow(Flow[ComplexState]):
    """Test flow with nested thread locks."""
    initial_state = ComplexState

    @start()
    async def step_1(self):
        with self.state.outer_lock:
            with self.state.inner.lock:
                self.state.value = "outer"
                self.state.inner.value = "inner"
                return "step 1"

    @listen(step_1)
    async def step_2(self, result):
        with self.state.outer_lock:
            with self.state.inner.lock:
                self.state.value += " -> outer 2"
                self.state.inner.value += " -> inner 2"
                return result + " -> step 2"


def test_flow_with_nested_locks():
    """Test Flow with nested thread locks in state."""
    flow = NestedLockFlow()
    result = asyncio.run(flow.kickoff_async())
    assert result == "step 1 -> step 2"
    assert flow.state.value == "outer -> outer 2"
    assert flow.state.inner.value == "inner -> inner 2"


class AsyncLockState(BaseModel):
    """Test state model with async locks."""
    model_config = {
        "arbitrary_types_allowed": True,
        "exclude": {"lock", "event"}
    }
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    lock: Optional[asyncio.Lock] = Field(default=None, exclude=True)
    event: Optional[asyncio.Event] = Field(default=None, exclude=True)
    value: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.lock is None:
            self.lock = asyncio.Lock()
        if self.event is None:
            self.event = asyncio.Event()


class AsyncLockFlow(Flow[AsyncLockState]):
    """Test flow with async locks."""
    initial_state = AsyncLockState

    @start()
    async def step_1(self):
        async with self.state.lock:
            self.state.value = "step 1"
            self.state.event.set()
            return "step 1"

    @listen(step_1)
    async def step_2(self, result):
        async with self.state.lock:
            await self.state.event.wait()
            self.state.value += " -> step 2"
            return result + " -> step 2"


def test_flow_with_async_locks():
    """Test Flow with async locks in state."""
    flow = AsyncLockFlow()
    result = asyncio.run(flow.kickoff_async())
    assert result == "step 1 -> step 2"
    assert flow.state.value == "step 1 -> step 2"
