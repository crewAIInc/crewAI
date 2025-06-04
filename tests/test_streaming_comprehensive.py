from unittest.mock import Mock, patch
from crewai import Agent, Task, Crew
from crewai.utilities.events.crew_events import CrewStreamChunkEvent
from crewai.utilities.events.llm_events import LLMStreamChunkEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus


def test_streaming_callback_called():
    """Test that streaming callback is called during execution."""
    callback_calls = []
    
    def stream_callback(chunk, agent_role, task_desc, step_type):
        callback_calls.append({
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
        
        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor._stream_callback = None
            mock_executor._task_description = None
            
            crew.kickoff(stream=True, stream_callback=stream_callback)
            
            assert hasattr(agent.agent_executor, '_stream_callback')
            assert agent.agent_executor._stream_callback == stream_callback
            assert hasattr(agent.agent_executor, '_task_description')
            assert agent.agent_executor._task_description == "Test task"


def test_crew_stream_chunk_event_creation():
    """Test CrewStreamChunkEvent can be created with all required fields."""
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


def test_streaming_disabled_by_default():
    """Test that streaming is disabled by default."""
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
        
        crew.kickoff()
        
        assert getattr(crew, '_stream_enabled', False) is False
        assert getattr(crew, '_stream_callback', None) is None


def test_streaming_parameters_propagation():
    """Test that streaming parameters are propagated through execution chain."""
    stream_callback = Mock()
    
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
        
        with patch.object(task, 'execute_sync') as mock_execute_sync:
            mock_execute_sync.return_value = Mock()
            
            crew.kickoff(stream=True, stream_callback=stream_callback)
            
            mock_execute_sync.assert_called_once()
            call_args = mock_execute_sync.call_args
            assert 'stream' in call_args.kwargs
            assert call_args.kwargs['stream'] is True
            assert 'stream_callback' in call_args.kwargs
            assert call_args.kwargs['stream_callback'] == stream_callback


def test_async_task_streaming():
    """Test that streaming works with async tasks."""
    stream_callback = Mock()
    
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
            agent=agent,
            async_execution=True
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )
        
        with patch.object(task, 'execute_async') as mock_execute_async:
            mock_future = Mock()
            mock_execute_async.return_value = mock_future
            
            crew.kickoff(stream=True, stream_callback=stream_callback)
            
            mock_execute_async.assert_called_once()
            call_args = mock_execute_async.call_args
            assert 'stream' in call_args.kwargs
            assert call_args.kwargs['stream'] is True
            assert 'stream_callback' in call_args.kwargs
            assert call_args.kwargs['stream_callback'] == stream_callback


def test_llm_stream_chunk_to_crew_stream_chunk():
    """Test that LLMStreamChunkEvent triggers CrewStreamChunkEvent."""
    received_crew_chunks = []
    
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(CrewStreamChunkEvent)
        def handle_crew_stream_chunk(source, event):
            received_crew_chunks.append(event)
        
        mock_source = Mock()
        mock_source.agent = Mock()
        mock_source.agent.role = "Test Agent"
        mock_source._task_description = "Test task"
        
        llm_event = LLMStreamChunkEvent(chunk="test chunk")
        
        from crewai.utilities.events.crewai_event_bus import crewai_event_bus
        crewai_event_bus.emit(mock_source, llm_event)
        
        assert len(received_crew_chunks) == 1
        crew_event = received_crew_chunks[0]
        assert crew_event.type == "crew_stream_chunk"
        assert crew_event.chunk == "test chunk"
        assert crew_event.agent_role == "Test Agent"
        assert crew_event.task_description == "Test task"
        assert crew_event.step_type == "llm_response"


def test_multiple_agents_streaming():
    """Test streaming with multiple agents."""
    stream_callback = Mock()
    
    with patch('crewai.llm.LLM') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "Agent response"
        mock_llm_class.return_value = mock_llm
        
        agent1 = Agent(
            role="Agent 1",
            goal="Goal 1",
            backstory="Backstory 1",
            llm=mock_llm,
            verbose=False
        )
        
        agent2 = Agent(
            role="Agent 2",
            goal="Goal 2",
            backstory="Backstory 2",
            llm=mock_llm,
            verbose=False
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
            verbose=False
        )
        
        crew.kickoff(stream=True, stream_callback=stream_callback)
        
        assert hasattr(crew, '_stream_enabled')
        assert crew._stream_enabled is True
        assert hasattr(crew, '_stream_callback')
        assert crew._stream_callback == stream_callback
