import threading
from unittest.mock import Mock, patch

import pytest

from crewai.agent import Agent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
)
from crewai.llm import LLM
from crewai.task import Task


@pytest.fixture
def base_agent():
    return Agent(
        role="test_agent",
        llm="gpt-4o-mini",
        goal="Test message_id",
        backstory="You are a test assistant",
    )


@pytest.fixture
def base_task(base_agent):
    return Task(
        description="Test message_id",
        expected_output="test",
        agent=base_agent,
    )


def test_llm_events_have_unique_message_ids_for_different_calls(base_agent, base_task):
    """Test that different LLM calls have different message_ids"""
    received_events = []
    event_received = threading.Event()
    
    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_llm_started(source, event):
        received_events.append(event)
        if len(received_events) >= 2:
            event_received.set()
    
    llm = LLM(model="gpt-4o-mini")
    
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Response 1", tool_calls=None))],
            usage=Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        
        llm.call("Test message 1", from_task=base_task, from_agent=base_agent)
        llm.call("Test message 2", from_task=base_task, from_agent=base_agent)
    
    assert event_received.wait(timeout=5), "Timeout waiting for LLM started events"
    assert len(received_events) >= 2
    assert received_events[0].message_id is not None
    assert received_events[1].message_id is not None
    assert received_events[0].message_id != received_events[1].message_id


def test_streaming_chunks_have_same_message_id(base_agent, base_task):
    """Test that all chunks from the same streaming call have the same message_id"""
    received_events = []
    lock = threading.Lock()
    all_events_received = threading.Event()
    
    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handle_stream_chunk(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) >= 3:
                all_events_received.set()
    
    llm = LLM(model="gpt-4o-mini", stream=True)
    
    def mock_stream_generator():
        yield Mock(
            choices=[Mock(delta=Mock(content="Hello", tool_calls=None))],
            usage=None,
        )
        yield Mock(
            choices=[Mock(delta=Mock(content=" ", tool_calls=None))],
            usage=None,
        )
        yield Mock(
            choices=[Mock(delta=Mock(content="World", tool_calls=None))],
            usage=Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
    
    with patch("litellm.completion", return_value=mock_stream_generator()):
        llm.call("Test streaming", from_task=base_task, from_agent=base_agent)
    
    assert all_events_received.wait(timeout=5), "Timeout waiting for stream chunk events"
    assert len(received_events) >= 3
    
    message_ids = [event.message_id for event in received_events]
    assert all(mid is not None for mid in message_ids)
    assert len(set(message_ids)) == 1, "All chunks should have the same message_id"


def test_completed_event_has_same_message_id_as_started(base_agent, base_task):
    """Test that Started and Completed events have the same message_id"""
    received_events = {"started": None, "completed": None}
    lock = threading.Lock()
    all_events_received = threading.Event()
    
    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_started(source, event):
        with lock:
            received_events["started"] = event
            if received_events["completed"] is not None:
                all_events_received.set()
    
    @crewai_event_bus.on(LLMCallCompletedEvent)
    def handle_completed(source, event):
        with lock:
            received_events["completed"] = event
            if received_events["started"] is not None:
                all_events_received.set()
    
    llm = LLM(model="gpt-4o-mini")
    
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Response", tool_calls=None))],
            usage=Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        
        llm.call("Test message", from_task=base_task, from_agent=base_agent)
    
    assert all_events_received.wait(timeout=5), "Timeout waiting for events"
    assert received_events["started"] is not None
    assert received_events["completed"] is not None
    assert received_events["started"].message_id is not None
    assert received_events["completed"].message_id is not None
    assert received_events["started"].message_id == received_events["completed"].message_id


def test_multiple_calls_same_agent_task_have_different_message_ids(base_agent, base_task):
    """Test that multiple calls from the same agent/task have different message_ids"""
    received_started_events = []
    lock = threading.Lock()
    all_events_received = threading.Event()
    
    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_started(source, event):
        with lock:
            received_started_events.append(event)
            if len(received_started_events) >= 3:
                all_events_received.set()
    
    llm = LLM(model="gpt-4o-mini")
    
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Response", tool_calls=None))],
            usage=Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        
        llm.call("Message 1", from_task=base_task, from_agent=base_agent)
        llm.call("Message 2", from_task=base_task, from_agent=base_agent)
        llm.call("Message 3", from_task=base_task, from_agent=base_agent)
    
    assert all_events_received.wait(timeout=5), "Timeout waiting for events"
    assert len(received_started_events) >= 3
    
    message_ids = [event.message_id for event in received_started_events]
    assert all(mid is not None for mid in message_ids)
    assert len(set(message_ids)) == 3, "Each call should have a unique message_id"
    
    task_ids = [event.task_id for event in received_started_events]
    agent_ids = [event.agent_id for event in received_started_events]
    assert len(set(task_ids)) == 1, "All calls should have the same task_id"
    assert len(set(agent_ids)) == 1, "All calls should have the same agent_id"
