from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_recording_mode():
    agent = Agent(
        role="Test Agent",
        goal="Test the recording functionality",
        backstory="A test agent for recording LLM responses",
    )
    
    task = Task(
        description="Return a simple response",
        expected_output="A simple response",
        agent=agent,
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        record_mode=True,
    )
    
    mock_handler = MagicMock()
    crew._llm_response_cache_handler = mock_handler
    
    mock_llm = MagicMock()
    agent.llm = mock_llm
    
    with patch('crewai.agent.Agent.execute_task', return_value="Test response"):
        with patch('crewai.utilities.llm_response_cache_handler.LLMResponseCacheHandler', return_value=mock_handler):
            crew.kickoff()
    
    mock_handler.start_recording.assert_called_once()
    
    mock_llm.set_response_cache_handler.assert_called_once_with(mock_handler)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_replay_mode():
    agent = Agent(
        role="Test Agent",
        goal="Test the replay functionality",
        backstory="A test agent for replaying LLM responses",
    )
    
    task = Task(
        description="Return a simple response",
        expected_output="A simple response",
        agent=agent,
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        replay_mode=True,
    )
    
    mock_handler = MagicMock()
    crew._llm_response_cache_handler = mock_handler
    
    mock_llm = MagicMock()
    agent.llm = mock_llm
    
    with patch('crewai.agent.Agent.execute_task', return_value="Test response"):
        with patch('crewai.utilities.llm_response_cache_handler.LLMResponseCacheHandler', return_value=mock_handler):
            crew.kickoff()
    
    mock_handler.start_replaying.assert_called_once()
    
    mock_llm.set_response_cache_handler.assert_called_once_with(mock_handler)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_record_replay_flags_conflict():
    with pytest.raises(ValueError):
        crew = Crew(
            agents=[],
            tasks=[],
            process=Process.sequential,
            record_mode=True,
            replay_mode=True,
        )
        crew.kickoff()
