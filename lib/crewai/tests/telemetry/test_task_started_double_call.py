"""Test for issue #3757 - task_started method calls _operation twice causing freezing."""

import threading
from unittest.mock import MagicMock, patch

import pytest
from crewai import Agent, Crew, Task
from crewai.telemetry import Telemetry


@pytest.fixture(autouse=True)
def cleanup_telemetry():
    """Clean up telemetry singleton between tests."""
    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()
    yield
    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()


@pytest.mark.telemetry
def test_task_started_does_not_call_operation_twice():
    """Test that task_started only calls _operation once when telemetry is enabled.
    
    This test verifies the fix for issue #3757 where task_started was calling
    _operation twice, causing the app to freeze when using CrewAI Tracing.
    """
    with patch("crewai.telemetry.telemetry.TracerProvider"):
        telemetry = Telemetry()
        telemetry.ready = True
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm="gpt-4o-mini",
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            name="TestCrew",
        )
        
        call_count = 0
        original_start_span = None
        
        def mock_start_span(name):
            """Mock start_span to count calls."""
            nonlocal call_count
            call_count += 1
            span = MagicMock()
            span.end = MagicMock()
            return span
        
        with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_tracer.start_span = mock_start_span
            mock_get_tracer.return_value = mock_tracer
            
            span = telemetry.task_started(crew, task)
            
            assert span is not None, "task_started should return a span when telemetry is enabled"
            assert call_count == 2, f"Expected 2 spans (Task Created + Task Execution), but got {call_count}"


@pytest.mark.telemetry
def test_task_started_returns_none_when_disabled():
    """Test that task_started returns None when telemetry is disabled."""
    with patch.dict("os.environ", {"CREWAI_DISABLE_TELEMETRY": "true"}):
        telemetry = Telemetry()
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm="gpt-4o-mini",
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            name="TestCrew",
        )
        
        span = telemetry.task_started(crew, task)
        
        assert span is None, "task_started should return None when telemetry is disabled"
