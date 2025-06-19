"""Tests for Agent fallback LLM functionality."""

import pytest
from unittest.mock import patch, MagicMock

from crewai import Agent, Task
from crewai.llm import LLM
from crewai.utilities.agent_utils import get_llm_response
from crewai.utilities import Printer
from litellm.exceptions import AuthenticationError, ContextWindowExceededError


def test_agent_with_fallback_llms_basic():
    """Test agent with fallback LLMs when primary fails."""
    primary_llm = LLM("gpt-4")
    fallback_llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="Test fallback functionality",
        backstory="I test fallback LLMs",
        llm=primary_llm,
        fallback_llms=[fallback_llm]
    )
    
    task = Task(
        description="Simple test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(primary_llm, 'call') as mock_primary, \
         patch.object(fallback_llm, 'call') as mock_fallback:
        
        mock_primary.side_effect = Exception("Primary LLM failed")
        mock_fallback.return_value = "Fallback response"
        
        result = agent.execute_task(task)
        
        assert result == "Fallback response"
        mock_primary.assert_called_once()
        mock_fallback.assert_called_once()


def test_agent_fallback_llms_multiple():
    """Test agent with multiple fallback LLMs."""
    primary_llm = LLM("gpt-4")
    fallback1 = LLM("gpt-3.5-turbo")
    fallback2 = LLM("claude-3-sonnet-20240229")
    
    agent = Agent(
        role="Test Agent",
        goal="Test multiple fallbacks",
        backstory="I test multiple fallback LLMs",
        llm=primary_llm,
        fallback_llms=[fallback1, fallback2]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(primary_llm, 'call') as mock_primary, \
         patch.object(fallback1, 'call') as mock_fallback1, \
         patch.object(fallback2, 'call') as mock_fallback2:
        
        mock_primary.side_effect = Exception("Primary failed")
        mock_fallback1.side_effect = Exception("Fallback 1 failed")
        mock_fallback2.return_value = "Fallback 2 response"
        
        result = agent.execute_task(task)
        
        assert result == "Fallback 2 response"
        mock_primary.assert_called_once()
        mock_fallback1.assert_called_once()
        mock_fallback2.assert_called_once()


def test_agent_fallback_auth_error_skips_fallbacks():
    """Test that authentication errors skip fallback attempts."""
    primary_llm = LLM("gpt-4")
    fallback_llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="Test auth error handling",
        backstory="I test auth error handling",
        llm=primary_llm,
        fallback_llms=[fallback_llm]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(primary_llm, 'call') as mock_primary, \
         patch.object(fallback_llm, 'call') as mock_fallback, \
         pytest.raises(AuthenticationError):
        
        mock_primary.side_effect = AuthenticationError(
            message="Invalid API key", llm_provider="openai", model="gpt-4"
        )
        
        agent.execute_task(task)
        
        mock_primary.assert_called_once()
        mock_fallback.assert_not_called()


def test_agent_fallback_context_window_error():
    """Test that context window errors try fallbacks."""
    primary_llm = LLM("gpt-4")
    fallback_llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="Test context window error handling",
        backstory="I test context window error handling",
        llm=primary_llm,
        fallback_llms=[fallback_llm]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(primary_llm, 'call') as mock_primary, \
         patch.object(fallback_llm, 'call') as mock_fallback:
        
        mock_primary.side_effect = ContextWindowExceededError(
            message="Context window exceeded", model="gpt-4", llm_provider="openai"
        )
        mock_fallback.return_value = "Fallback response"
        
        result = agent.execute_task(task)
        
        assert result == "Fallback response"
        mock_primary.assert_called_once()
        mock_fallback.assert_called_once()


def test_agent_all_llms_fail():
    """Test behavior when all LLMs fail."""
    primary_llm = LLM("gpt-4")
    fallback_llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="Test all LLMs failing",
        backstory="I test all LLMs failing",
        llm=primary_llm,
        fallback_llms=[fallback_llm]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(primary_llm, 'call') as mock_primary, \
         patch.object(fallback_llm, 'call') as mock_fallback, \
         pytest.raises(Exception, match="Fallback failed"):
        
        mock_primary.side_effect = Exception("Primary failed")
        mock_fallback.side_effect = Exception("Fallback failed")
        
        agent.execute_task(task)
        
        mock_primary.assert_called_once()
        mock_fallback.assert_called_once()


def test_agent_backward_compatibility():
    """Test that agents without fallback LLMs work as before."""
    agent = Agent(
        role="Test Agent",
        goal="Test backward compatibility",
        backstory="I test backward compatibility",
        llm=LLM("gpt-4")
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(agent.llm, 'call') as mock_llm:
        mock_llm.return_value = "Primary response"
        
        result = agent.execute_task(task)
        
        assert result == "Primary response"
        mock_llm.assert_called_once()


def test_get_llm_response_with_fallbacks():
    """Test get_llm_response function directly with fallbacks."""
    primary_llm = MagicMock()
    fallback_llm = MagicMock()
    printer = Printer()
    
    primary_llm.call.side_effect = Exception("Primary failed")
    fallback_llm.call.return_value = "Fallback success"
    
    result = get_llm_response(
        llm=primary_llm,
        messages=[{"role": "user", "content": "test"}],
        callbacks=[],
        printer=printer,
        fallback_llms=[fallback_llm]
    )
    
    assert result == "Fallback success"
    primary_llm.call.assert_called_once()
    fallback_llm.call.assert_called_once()


def test_get_llm_response_no_fallbacks():
    """Test get_llm_response function without fallbacks (backward compatibility)."""
    primary_llm = MagicMock()
    printer = Printer()
    
    primary_llm.call.return_value = "Primary success"
    
    result = get_llm_response(
        llm=primary_llm,
        messages=[{"role": "user", "content": "test"}],
        callbacks=[],
        printer=printer
    )
    
    assert result == "Primary success"
    primary_llm.call.assert_called_once()


def test_agent_fallback_llms_string_initialization():
    """Test that fallback LLMs can be initialized with string model names."""
    agent = Agent(
        role="Test Agent",
        goal="Test string initialization",
        backstory="I test string initialization",
        llm="gpt-4",
        fallback_llms=["gpt-3.5-turbo", "claude-3-sonnet-20240229"]
    )
    
    assert agent.fallback_llms is not None
    assert len(agent.fallback_llms) == 2
    assert hasattr(agent.fallback_llms[0], 'call')
    assert hasattr(agent.fallback_llms[1], 'call')


def test_agent_primary_success_no_fallback():
    """Test that fallback LLMs are not called when primary succeeds."""
    primary_llm = LLM("gpt-4")
    fallback_llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="Test primary success",
        backstory="I test primary success",
        llm=primary_llm,
        fallback_llms=[fallback_llm]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(primary_llm, 'call') as mock_primary, \
         patch.object(fallback_llm, 'call') as mock_fallback:
        
        mock_primary.return_value = "Primary success"
        
        result = agent.execute_task(task)
        
        assert result == "Primary success"
        mock_primary.assert_called_once()
        mock_fallback.assert_not_called()


def test_agent_empty_response_triggers_fallback():
    """Test that empty responses from primary LLM trigger fallback."""
    primary_llm = LLM("gpt-4")
    fallback_llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="Test empty response handling",
        backstory="I test empty response handling",
        llm=primary_llm,
        fallback_llms=[fallback_llm]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    with patch.object(primary_llm, 'call') as mock_primary, \
         patch.object(fallback_llm, 'call') as mock_fallback:
        
        mock_primary.return_value = ""
        mock_fallback.return_value = "Fallback response"
        
        result = agent.execute_task(task)
        
        assert result == "Fallback response"
        mock_primary.assert_called_once()
        mock_fallback.assert_called_once()
