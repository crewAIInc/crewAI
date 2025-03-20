"""Test Gemini models with HTML templates."""

from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Task
from crewai.llm import LLM


def test_gemini_empty_response_handling():
    """Test that empty responses from Gemini models are handled correctly."""
    # Create a mock LLM instance
    llm = LLM(model="gemini/gemini-pro")
    
    # Create a mock response with empty content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = ""
    
    # Mock litellm.completion to return our mock response
    with patch('litellm.completion', return_value=mock_response):
        # Call the non-streaming response handler directly
        result = llm._handle_non_streaming_response({"model": "gemini/gemini-pro"})
        
        # Verify that our fix works - empty string should be replaced with placeholder
        assert "Response processed successfully" in result
        assert "HTML template" in result


def test_openrouter_gemini_empty_response_handling():
    """Test that empty responses from OpenRouter with Gemini models are handled correctly."""
    # Create a mock LLM instance with OpenRouter base URL
    llm = LLM(
        model="openrouter/google/gemini-pro", 
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Create a mock response with empty content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = ""
    
    # Mock litellm.completion to return our mock response
    with patch('litellm.completion', return_value=mock_response):
        # Call the non-streaming response handler directly
        result = llm._handle_non_streaming_response({"model": "openrouter/google/gemini-pro"})
        
        # Verify that our fix works - empty string should be replaced with placeholder
        assert "Response processed successfully" in result
        assert "HTML template" in result


def test_gemini_none_response_handling():
    """Test that None responses are properly handled."""
    llm = LLM(model="gemini/gemini-pro")
    
    # Create a mock response with None content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = None
    
    # Mock litellm.completion to return our mock response
    with patch('litellm.completion', return_value=mock_response):
        # Call the non-streaming response handler directly
        # None content should be converted to empty string and then handled
        result = llm._handle_non_streaming_response({"model": "gemini/gemini-pro"})
        
        # Verify that our fix works - None should be converted to empty string
        # and then handled as an empty string for Gemini models
        assert "Response processed successfully" in result
        assert "HTML template" in result


@pytest.mark.parametrize("model_name,base_url", [
    ("gemini/gemini-pro", None),
    ("gemini-pro", None),
    ("google/gemini-pro", None),
    ("openrouter/google/gemini-pro", "https://openrouter.ai/api/v1"),
    ("openrouter/gemini-pro", "https://openrouter.ai/api/v1"),
])
def test_various_gemini_configurations(model_name, base_url):
    """Test different Gemini model configurations with the _is_gemini_model helper."""
    # Create a mock LLM instance with the specified model and base URL
    llm = LLM(model=model_name, base_url=base_url)
    
    # Verify that _is_gemini_model correctly identifies all these configurations
    assert llm._is_gemini_model() is True
    
    # Create a mock response with empty content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = ""
    
    # Mock litellm.completion to return our mock response
    with patch('litellm.completion', return_value=mock_response):
        # Call the non-streaming response handler directly
        result = llm._handle_non_streaming_response({"model": model_name})
        
        # Verify that our fix works for all Gemini configurations
        assert "Response processed successfully" in result
        assert "HTML template" in result


def test_non_gemini_model():
    """Test that non-Gemini models don't get special handling for empty responses."""
    # Create a mock LLM instance with a non-Gemini model
    llm = LLM(model="gpt-4")
    
    # Verify that _is_gemini_model correctly identifies this as not a Gemini model
    assert llm._is_gemini_model() is False
    
    # Create a mock response with empty content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = ""
    
    # Mock litellm.completion to return our mock response
    with patch('litellm.completion', return_value=mock_response):
        # Call the non-streaming response handler directly
        result = llm._handle_non_streaming_response({"model": "gpt-4"})
        
        # Verify that non-Gemini models just return the empty string
        assert result == ""
