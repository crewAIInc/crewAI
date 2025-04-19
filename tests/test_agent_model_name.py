import pytest
from unittest.mock import MagicMock, patch
from crewai import Agent
from crewai.llm import LLM


def test_normalize_model_name_method():
    """Test that the _normalize_model_name method correctly handles model names with 'models/' prefix"""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm="gpt-4"
    )
    
    model_with_prefix = "models/gemini/gemini-1.5-flash"
    normalized_name = agent._normalize_model_name(model_with_prefix)
    assert normalized_name == "gemini/gemini-1.5-flash"
    
    regular_model = "gpt-4"
    assert agent._normalize_model_name(regular_model) == "gpt-4"
    
    assert agent._normalize_model_name(None) is None
    
    assert agent._normalize_model_name(123) == 123


def test_agent_with_regular_model_name():
    """Test that the Agent class doesn't modify normal model names"""
    with patch('crewai.agent.LLM') as mock_llm:
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm="gpt-4"
        )
        
        args, kwargs = mock_llm.call_args
        assert kwargs["model"] == "gpt-4"
