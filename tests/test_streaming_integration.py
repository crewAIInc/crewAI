from unittest.mock import Mock, patch
from crewai import Agent, Task, Crew
from crewai.utilities.events.crew_events import CrewStreamChunkEvent
from crewai.utilities.events.llm_events import LLMStreamChunkEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus


def test_streaming_callback_integration():
    """Test that streaming callback is properly integrated through the execution chain."""
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
        
        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor._stream_callback = None
            mock_executor._task_description = None
            
            crew.kickoff(stream=True, stream_callback=stream_callback)
            
            assert hasattr(agent.agent_executor, '_stream_callback')
            assert hasattr(agent.agent_executor, '_task_description')


def test_crew_stream_chunk_event_emission():
    """Test that CrewStreamChunkEvent is emitted when LLMStreamChunkEvent occurs."""
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
        
        from crewai.utilities.events.event_listener import event_listener
        event_listener.on_llm_stream_chunk(mock_source, llm_event)
        
        assert len(received_crew_chunks) == 1
        crew_event = received_crew_chunks[0]
        assert crew_event.type == "crew_stream_chunk"
        assert crew_event.chunk == "test chunk"
        assert crew_event.agent_role == "Test Agent"
        assert crew_event.task_description == "Test task"
        assert crew_event.step_type == "llm_response"


def test_streaming_with_multiple_agents():
    """Test streaming works correctly with multiple agents."""
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
        mock_llm.call.return_value = "Agent response"
        mock_llm.supports_stop_words = True
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
    """Test that streaming parameters are properly propagated through execution chain."""
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
