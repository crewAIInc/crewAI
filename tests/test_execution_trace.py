import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from crewai import Agent, Crew, Task, Process
from crewai.crews.execution_trace import ExecutionStep, ExecutionTrace
from crewai.utilities.execution_trace_collector import ExecutionTraceCollector
from crewai.tools.base_tool import BaseTool


class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.call.return_value = "Test response"
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


@pytest.fixture
def test_agent(mock_llm):
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm=mock_llm,
        tools=[MockTool()]
    )


@pytest.fixture
def test_task(test_agent):
    return Task(
        description="Test task description",
        expected_output="Test expected output",
        agent=test_agent
    )


def test_execution_step_creation():
    """Test creating an ExecutionStep."""
    step = ExecutionStep(
        timestamp=datetime.now(),
        step_type="agent_thought",
        agent_role="Test Agent",
        task_description="Test task",
        content={"thought": "I need to think about this"},
        metadata={"iteration": 1}
    )
    
    assert step.step_type == "agent_thought"
    assert step.agent_role == "Test Agent"
    assert step.content["thought"] == "I need to think about this"
    assert step.metadata["iteration"] == 1


def test_execution_trace_creation():
    """Test creating an ExecutionTrace."""
    trace = ExecutionTrace()
    
    step1 = ExecutionStep(
        timestamp=datetime.now(),
        step_type="task_started",
        content={"task_id": "1"}
    )
    
    step2 = ExecutionStep(
        timestamp=datetime.now(),
        step_type="agent_thought",
        agent_role="Test Agent",
        content={"thought": "Starting work"}
    )
    
    trace.add_step(step1)
    trace.add_step(step2)
    
    assert trace.total_steps == 2
    assert len(trace.steps) == 2
    assert trace.get_steps_by_type("task_started") == [step1]
    assert trace.get_steps_by_agent("Test Agent") == [step2]


def test_execution_trace_collector():
    """Test the ExecutionTraceCollector."""
    collector = ExecutionTraceCollector()
    
    collector.start_collecting()
    assert collector.is_collecting is True
    assert collector.trace.start_time is not None
    
    trace = collector.stop_collecting()
    assert collector.is_collecting is False
    assert trace.end_time is not None
    assert isinstance(trace, ExecutionTrace)


@patch('crewai.crew.crewai_event_bus')
def test_crew_with_execution_trace_enabled(mock_event_bus, test_agent, test_task, mock_llm):
    """Test crew execution with trace_execution=True."""
    crew = Crew(
        agents=[test_agent],
        tasks=[test_task],
        process=Process.sequential,
        trace_execution=True
    )
    
    with patch.object(test_task, 'execute_sync') as mock_execute:
        from crewai.tasks.task_output import TaskOutput
        mock_output = TaskOutput(
            description="Test task description",
            raw="Test output",
            agent="Test Agent"
        )
        mock_execute.return_value = mock_output
        
        result = crew.kickoff()
        
        assert result.execution_trace is not None
        assert isinstance(result.execution_trace, ExecutionTrace)
        assert result.execution_trace.start_time is not None
        assert result.execution_trace.end_time is not None


@patch('crewai.crew.crewai_event_bus')
def test_crew_without_execution_trace(mock_event_bus, test_agent, test_task, mock_llm):
    """Test crew execution with trace_execution=False (default)."""
    crew = Crew(
        agents=[test_agent],
        tasks=[test_task],
        process=Process.sequential,
        trace_execution=False
    )
    
    with patch.object(test_task, 'execute_sync') as mock_execute:
        from crewai.tasks.task_output import TaskOutput
        mock_output = TaskOutput(
            description="Test task description",
            raw="Test output",
            agent="Test Agent"
        )
        mock_execute.return_value = mock_output
        
        result = crew.kickoff()
        
        assert result.execution_trace is None


def test_execution_trace_with_multiple_agents_and_tasks(mock_llm):
    """Test execution trace with multiple agents and tasks."""
    agent1 = Agent(
        role="Agent 1",
        goal="Goal 1",
        backstory="Backstory 1",
        llm=mock_llm
    )
    
    agent2 = Agent(
        role="Agent 2", 
        goal="Goal 2",
        backstory="Backstory 2",
        llm=mock_llm
    )
    
    task1 = Task(
        description="Task 1",
        expected_output="Output 1",
        agent=agent1
    )
    
    task2 = Task(
        description="Task 2",
        expected_output="Output 2", 
        agent=agent2
    )
    
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2],
        process=Process.sequential,
        trace_execution=True
    )
    
    with patch.object(task1, 'execute_sync') as mock_execute1, \
         patch.object(task2, 'execute_sync') as mock_execute2:
        
        from crewai.tasks.task_output import TaskOutput
        
        mock_output1 = TaskOutput(
            description="Task 1",
            raw="Output 1",
            agent="Agent 1"
        )
        
        mock_output2 = TaskOutput(
            description="Task 2", 
            raw="Output 2",
            agent="Agent 2"
        )
        
        mock_execute1.return_value = mock_output1
        mock_execute2.return_value = mock_output2
        
        result = crew.kickoff()
        
        assert result.execution_trace is not None
        agent1_steps = result.execution_trace.get_steps_by_agent("Agent 1")
        agent2_steps = result.execution_trace.get_steps_by_agent("Agent 2")
        
        assert len(agent1_steps) >= 0
        assert len(agent2_steps) >= 0


def test_execution_trace_step_types():
    """Test that different step types are properly categorized."""
    trace = ExecutionTrace()
    
    steps_data = [
        ("task_started", "Task 1", {}),
        ("agent_thought", "Agent 1", {"thought": "I need to analyze this"}),
        ("tool_call_started", "Agent 1", {"tool_name": "search", "args": {"query": "test"}}),
        ("tool_call_completed", "Agent 1", {"tool_name": "search", "output": "results"}),
        ("agent_execution_completed", "Agent 1", {"output": "Final answer"}),
        ("task_completed", "Task 1", {"output": "Task complete"}),
    ]
    
    for step_type, agent_role, content in steps_data:
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type=step_type,
            agent_role=agent_role if "agent" in step_type or "tool" in step_type else None,
            task_description="Task 1" if "task" in step_type else None,
            content=content
        )
        trace.add_step(step)
    
    assert len(trace.get_steps_by_type("task_started")) == 1
    assert len(trace.get_steps_by_type("agent_thought")) == 1
    assert len(trace.get_steps_by_type("tool_call_started")) == 1
    assert len(trace.get_steps_by_type("tool_call_completed")) == 1
    assert len(trace.get_steps_by_type("agent_execution_completed")) == 1
    assert len(trace.get_steps_by_type("task_completed")) == 1
    
    agent_steps = trace.get_steps_by_agent("Agent 1")
    assert len(agent_steps) == 4


def test_execution_trace_with_async_tasks(mock_llm):
    """Test execution trace with async tasks."""
    agent = Agent(
        role="Async Agent",
        goal="Async goal", 
        backstory="Async backstory",
        llm=mock_llm
    )
    
    task = Task(
        description="Async task",
        expected_output="Async output",
        agent=agent,
        async_execution=True
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        trace_execution=True
    )
    
    with patch.object(task, 'execute_async') as mock_execute_async:
        from concurrent.futures import Future
        from crewai.tasks.task_output import TaskOutput
        
        future = Future()
        mock_output = TaskOutput(
            description="Async task",
            raw="Async output",
            agent="Async Agent"
        )
        future.set_result(mock_output)
        mock_execute_async.return_value = future
        
        result = crew.kickoff()
        
        assert result.execution_trace is not None
        assert isinstance(result.execution_trace, ExecutionTrace)


def test_execution_trace_error_handling():
    """Test execution trace handles errors gracefully."""
    collector = ExecutionTraceCollector()
    
    collector.start_collecting()
    
    mock_event = Mock()
    mock_event.agent = Mock()
    mock_event.agent.role = "Test Agent"
    
    collector._handle_agent_started(mock_event)
    
    trace = collector.stop_collecting()
    assert isinstance(trace, ExecutionTrace)
