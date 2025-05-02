from unittest.mock import patch, MagicMock

import pytest

from crewai.llm import LLM


@patch('crewai.llm.LLM._prepare_completion_params')
def test_empty_messages_validation(mock_prepare):
    """
    Test that LLM.call() raises a ValueError when an empty messages list is passed.
    This prevents the IndexError in LiteLLM's ollama_pt() function.
    """
    llm = LLM(model="gpt-3.5-turbo")  # Any model will do for this test
    
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.call(messages=[])
    
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.call(messages=None)
    
    mock_prepare.assert_not_called()


@patch('crewai.llm.LLM._prepare_completion_params')
def test_empty_string_message(mock_prepare):
    """
    Test that LLM.call() raises a ValueError when an empty string message is passed.
    """
    llm = LLM(model="gpt-3.5-turbo")
    
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.call(messages="")
    
    mock_prepare.assert_not_called()


@patch('crewai.llm.LLM._prepare_completion_params')
def test_invalid_message_format(mock_prepare):
    """
    Test that LLM.call() raises a TypeError when a message with invalid format is passed.
    """
    mock_prepare.side_effect = TypeError("Invalid message format")
    llm = LLM(model="gpt-3.5-turbo")
    
    with pytest.raises(TypeError, match="Invalid message format"):
        llm.call(messages=[{}])


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ollama_model_empty_messages():
    """
    Test that LLM.call() with an Ollama model raises a ValueError 
    when an empty messages list is passed.
    """
    llm = LLM(model="ollama/llama3")
    
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.call(messages=[])
