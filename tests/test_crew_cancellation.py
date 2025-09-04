import threading
import time
from unittest.mock import Mock, patch
import pytest
from crewai import Agent, Crew, Task
from crewai.process import Process
from crewai.events.types.crew_events import CrewKickoffCancelledEvent
from crewai.tasks.task_output import TaskOutput


@pytest.fixture
def mock_agent():
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        verbose=False,
    )


@pytest.fixture
def slow_task():
    """A task that takes some time to complete for testing cancellation"""
    def slow_execution(*args, **kwargs):
        time.sleep(0.5)
        return TaskOutput(
            description="Task completed",
            raw="Task completed",
            agent="Test Agent"
        )
    
    task = Task(
        description="A slow task for testing",
        expected_output="Task output",
    )
    task.execute_sync = Mock(side_effect=slow_execution)
    return task


def test_crew_cancellation_basic(mock_agent, slow_task):
    """Test basic cancellation functionality"""
    crew = Crew(agents=[mock_agent], tasks=[slow_task], verbose=False)
    
    assert not crew.is_cancelled()
    
    crew.cancel()
    assert crew.is_cancelled()


def test_crew_cancellation_during_execution(mock_agent):
    """Test cancellation during crew execution"""
    tasks = []
    for i in range(3):
        task = Task(
            description=f"Task {i}",
            expected_output="Output",
        )
        task.execute_sync = Mock(return_value=TaskOutput(
            description=f"Task {i} completed",
            raw=f"Output {i}",
            agent="Test Agent"
        ))
        tasks.append(task)
    
    crew = Crew(agents=[mock_agent], tasks=tasks, verbose=False)
    
    result = None
    exception = None
    
    def run_crew():
        nonlocal result, exception
        try:
            result = crew.kickoff()
        except Exception as e:
            exception = e
    
    thread = threading.Thread(target=run_crew)
    thread.start()
    
    time.sleep(0.1)
    crew.cancel()
    
    thread.join(timeout=2)
    
    assert crew.is_cancelled()
    assert result is not None
    assert exception is None


def test_crew_cancellation_events(mock_agent, slow_task):
    """Test that cancellation events are emitted properly"""
    crew = Crew(agents=[mock_agent], tasks=[slow_task], verbose=False)
    
    with patch('crewai.events.event_bus.crewai_event_bus.emit') as mock_emit:
        crew.cancel()
        result = crew.kickoff()
        
        cancellation_events = [
            call for call in mock_emit.call_args_list
            if len(call[0]) > 1 and isinstance(call[0][1], CrewKickoffCancelledEvent)
        ]
        assert len(cancellation_events) > 0


def test_crew_reuse_after_cancellation(mock_agent):
    """Test that crew can be reused after cancellation"""
    task = Task(
        description="Test task",
        expected_output="Test output",
    )
    task.execute_sync = Mock(return_value=TaskOutput(
        description="Task completed",
        raw="Task completed",
        agent="Test Agent"
    ))
    
    crew = Crew(agents=[mock_agent], tasks=[task], verbose=False)
    
    crew.cancel()
    result1 = crew.kickoff()
    
    result2 = crew.kickoff()
    assert not crew.is_cancelled()


def test_crew_cancellation_hierarchical_process(mock_agent):
    """Test cancellation works with hierarchical process"""
    task = Task(
        description="Test task",
        expected_output="Test output",
    )
    task.execute_sync = Mock(return_value=TaskOutput(
        description="Task completed",
        raw="Task completed",
        agent="Test Agent"
    ))
    
    crew = Crew(
        agents=[mock_agent], 
        tasks=[task], 
        process=Process.hierarchical,
        manager_llm="gpt-3.5-turbo",
        verbose=False
    )
    
    crew.cancel()
    result = crew.kickoff()
    assert crew.is_cancelled()


def test_crew_cancellation_thread_safety():
    """Test thread safety of cancellation mechanism"""
    agent = Agent(role="Test", goal="Test", backstory="Test", verbose=False)
    task = Task(description="Test", expected_output="Test")
    task.execute_sync = Mock(return_value=TaskOutput(
        description="Task completed",
        raw="Task completed",
        agent="Test Agent"
    ))
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    
    def toggle_cancellation():
        for _ in range(100):
            crew.cancel()
            crew._reset_cancellation()
    
    threads = [threading.Thread(target=toggle_cancellation) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    assert isinstance(crew.is_cancelled(), bool)


def test_crew_cancellation_with_async_tasks(mock_agent):
    """Test cancellation with async tasks"""
    task = Task(
        description="Async test task",
        expected_output="Test output",
        async_execution=True
    )
    
    def mock_execute_async(*args, **kwargs):
        from concurrent.futures import Future
        future = Future()
        future.set_result(TaskOutput(
            description="Async task completed",
            raw="Async task completed",
            agent="Test Agent"
        ))
        return future
    
    task.execute_async = Mock(side_effect=mock_execute_async)
    
    crew = Crew(agents=[mock_agent], tasks=[task], verbose=False)
    
    crew.cancel()
    result = crew.kickoff()
    assert crew.is_cancelled()


def test_crew_cancellation_partial_results(mock_agent):
    """Test that partial results are returned when cancelled"""
    tasks = []
    for i in range(3):
        task = Task(
            description=f"Task {i}",
            expected_output="Output",
        )
        task.execute_sync = Mock(return_value=TaskOutput(
            description=f"Task {i} completed",
            raw=f"Output {i}",
            agent="Test Agent"
        ))
        tasks.append(task)
    
    crew = Crew(agents=[mock_agent], tasks=tasks, verbose=False)
    
    crew.cancel()
    result = crew.kickoff()
    
    assert result is not None
    assert hasattr(result, 'tasks_output')
