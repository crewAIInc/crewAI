import pytest
from unittest.mock import Mock, patch
from crewai import Agent, Task, Crew
from crewai.utilities.events.crew_events import CrewStreamChunkEvent, TaskStreamChunkEvent, AgentStreamChunkEvent


@pytest.fixture
def mock_llm():
    return Mock()


@pytest.fixture
def agent(mock_llm):
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm=mock_llm,
        verbose=False
    )


@pytest.fixture
def task(agent):
    return Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )


@pytest.fixture
def crew(agent, task):
    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )


def test_crew_streaming_enabled():
    """Test that crew streaming can be enabled."""
    received_chunks = []
    
    def stream_callback(chunk, agent_role, task_desc, step_type):
        received_chunks.append({
            'chunk': chunk,
            'agent_role': agent_role,
            'task_desc': task_desc,
            'step_type': step_type
        })
    
    with patch('crewai.llm.LLM') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "Test response"
        mock_llm.supports_stop_words = True
        mock_llm_class.return_value = mock_llm
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal", 
            backstory="Test backstory",
            llm=mock_llm,
            verbose=False
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )
        
        with patch.object(crew, '_execute_tasks') as mock_execute:
            mock_execute.return_value = Mock()
            
            crew.kickoff(stream=True, stream_callback=stream_callback)
            
            assert hasattr(crew, '_stream_enabled')
            assert crew._stream_enabled is True
            assert hasattr(crew, '_stream_callback')
            assert crew._stream_callback == stream_callback


def test_crew_streaming_disabled_by_default():
    """Test that crew streaming is disabled by default."""
    with patch('crewai.llm.LLM') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "Test response"
        mock_llm.supports_stop_words = True
        mock_llm_class.return_value = mock_llm
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory", 
            llm=mock_llm,
            verbose=False
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )
        
        with patch.object(crew, '_execute_tasks') as mock_execute:
            mock_execute.return_value = Mock()
            
            crew.kickoff()
            
            assert getattr(crew, '_stream_enabled', False) is False
            assert getattr(crew, '_stream_callback', None) is None


def test_crew_stream_chunk_event():
    """Test CrewStreamChunkEvent creation and properties."""
    event = CrewStreamChunkEvent(
        chunk="test chunk",
        agent_role="Test Agent",
        task_description="Test task",
        step_type="agent_thinking",
        crew=None,
        crew_name="TestCrew"
    )
    
    assert event.type == "crew_stream_chunk"
    assert event.chunk == "test chunk"
    assert event.agent_role == "Test Agent"
    assert event.task_description == "Test task"
    assert event.step_type == "agent_thinking"


def test_task_stream_chunk_event():
    """Test TaskStreamChunkEvent creation and properties."""
    event = TaskStreamChunkEvent(
        chunk="test chunk",
        task_description="Test task",
        agent_role="Test Agent",
        step_type="task_execution"
    )
    
    assert event.type == "task_stream_chunk"
    assert event.chunk == "test chunk"
    assert event.task_description == "Test task"
    assert event.agent_role == "Test Agent"
    assert event.step_type == "task_execution"


def test_agent_stream_chunk_event():
    """Test AgentStreamChunkEvent creation and properties."""
    event = AgentStreamChunkEvent(
        chunk="test chunk",
        agent_role="Test Agent",
        step_type="agent_thinking"
    )
    
    assert event.type == "agent_stream_chunk"
    assert event.chunk == "test chunk"
    assert event.agent_role == "Test Agent"
    assert event.step_type == "agent_thinking"


def test_streaming_integration_with_llm():
    """Test that streaming integrates with existing LLM streaming."""
    received_callback_chunks = []
    
    def stream_callback(chunk, agent_role, task_desc, step_type):
        received_callback_chunks.append({
            'chunk': chunk,
            'agent_role': agent_role,
            'task_desc': task_desc,
            'step_type': step_type
        })
    
    with patch('crewai.llm.LLM') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "Here's a joke: Why did the robot cross the road? To get to the other side!"
        mock_llm_class.return_value = mock_llm
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm=mock_llm,
            verbose=False
        )
        
        task = Task(
            description="Tell me a short joke",
            expected_output="A short joke",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )
        
        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor._stream_callback = None
            mock_executor._task_description = None
            
            crew.kickoff(stream=True, stream_callback=stream_callback)
            
            assert hasattr(agent.agent_executor, '_stream_callback')
            assert hasattr(agent.agent_executor, '_task_description')


def test_streaming_parameters_propagation():
    """Test that streaming parameters are properly propagated through the execution chain."""
    with patch('crewai.llm.LLM') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "Test response"
        mock_llm.supports_stop_words = True
        mock_llm_class.return_value = mock_llm
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm=mock_llm,
            verbose=False
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )
        
        stream_callback = Mock()
        
        with patch.object(task, 'execute_sync') as mock_execute_sync:
            mock_execute_sync.return_value = Mock()
            
            crew.kickoff(stream=True, stream_callback=stream_callback)
            
            mock_execute_sync.assert_called_once()
            call_args = mock_execute_sync.call_args
            assert 'stream' in call_args.kwargs
            assert call_args.kwargs['stream'] is True
            assert 'stream_callback' in call_args.kwargs
            assert call_args.kwargs['stream_callback'] == stream_callback
