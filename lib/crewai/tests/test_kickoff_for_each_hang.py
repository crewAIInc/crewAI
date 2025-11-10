"""Test for issue #3871: kickoff_for_each() hang fix."""
import os
import threading
import time
from unittest.mock import Mock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.crew_events import CrewKickoffCompletedEvent


@pytest.fixture
def simple_crew():
    """Create a simple crew for testing."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        verbose=False,
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )
    
    return crew


def test_kickoff_for_each_waits_for_event_handlers(simple_crew):
    """Test that kickoff_for_each waits for event handlers to complete.
    
    This test verifies the fix for issue #3871 by registering a slow
    sync handler and ensuring kickoff_for_each waits for it to complete.
    """
    handler_completed = threading.Event()
    handler_call_count = 0
    
    def slow_handler(source, event):
        nonlocal handler_call_count
        handler_call_count += 1
        time.sleep(0.1)  # Simulate slow handler
        handler_completed.set()
    
    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.register_handler(
            CrewKickoffCompletedEvent,
            slow_handler,
        )
        
        # Mock the task execution to avoid actual LLM calls
        with patch.object(simple_crew, '_run_sequential_process') as mock_run:
            mock_output = Mock()
            mock_output.raw = "Test output"
            mock_output.pydantic = None
            mock_output.json_dict = None
            mock_run.return_value = Mock(
                raw="Test output",
                pydantic=None,
                json_dict=None,
                tasks_output=[mock_output],
                token_usage=Mock(total_tokens=0),
            )
            
            start_time = time.time()
            results = simple_crew.kickoff_for_each(
                inputs=[{"test": "input1"}, {"test": "input2"}]
            )
            elapsed_time = time.time() - start_time
            
            # Verify results were returned
            assert len(results) == 2
            
            # Verify handler was called for each kickoff
            assert handler_call_count == 2
            
            # Verify the execution waited for handlers (should take at least 0.2s for 2 handlers)
            assert elapsed_time >= 0.2, (
                f"kickoff_for_each returned too quickly ({elapsed_time:.3f}s), "
                "suggesting it didn't wait for event handlers"
            )
            
            # Verify handler completed
            assert handler_completed.is_set()


def test_kickoff_waits_for_event_handlers_on_error(simple_crew):
    """Test that kickoff waits for event handlers even when an error occurs."""
    handler_completed = threading.Event()
    
    def error_handler(source, event):
        time.sleep(0.1)  # Simulate slow handler
        handler_completed.set()
    
    with crewai_event_bus.scoped_handlers():
        from crewai.events.types.crew_events import CrewKickoffFailedEvent
        crewai_event_bus.register_handler(
            CrewKickoffFailedEvent,
            error_handler,
        )
        
        # Mock the task execution to raise an error
        with patch.object(simple_crew, '_run_sequential_process') as mock_run:
            mock_run.side_effect = RuntimeError("Test error")
            
            start_time = time.time()
            with pytest.raises(RuntimeError, match="Test error"):
                simple_crew.kickoff()
            elapsed_time = time.time() - start_time
            
            # Verify the execution waited for handlers (should take at least 0.1s)
            assert elapsed_time >= 0.1, (
                f"kickoff returned too quickly ({elapsed_time:.3f}s), "
                "suggesting it didn't wait for error event handlers"
            )
            
            # Verify handler completed
            assert handler_completed.is_set()


def test_tracing_disabled_flag_respected():
    """Test that CREWAI_DISABLE_TRACING flag prevents tracing setup."""
    from crewai.events.listeners.tracing.utils import is_tracing_disabled
    
    # Test with CREWAI_DISABLE_TRACING=true
    with patch.dict(os.environ, {"CREWAI_DISABLE_TRACING": "true"}):
        assert is_tracing_disabled() is True
    
    # Test with OTEL_SDK_DISABLED=true
    with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
        assert is_tracing_disabled() is True
    
    # Test with CREWAI_DISABLE_TRACKING=true
    with patch.dict(os.environ, {"CREWAI_DISABLE_TRACKING": "true"}):
        assert is_tracing_disabled() is True
    
    # Test with no disable flags
    with patch.dict(os.environ, {}, clear=True):
        assert is_tracing_disabled() is False


def test_tracing_not_enabled_when_disabled_flag_set():
    """Test that tracing is not enabled when disable flag is set."""
    from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
    
    # Mock TraceCollectionListener.setup_listeners to track if it's called
    with patch.object(TraceCollectionListener, 'setup_listeners') as mock_setup:
        with patch.dict(os.environ, {
            "CREWAI_DISABLE_TRACING": "true",
            "CREWAI_TESTING": "true",  # Prevent first-time auto-collection
        }):
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                verbose=False,
            )
            
            task = Task(
                description="Test task",
                expected_output="Test output",
                agent=agent,
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False,
            )
            
            # Verify setup_listeners was not called
            mock_setup.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
