"""Test Gemini models with HTML templates."""

import pytest
from unittest.mock import patch, MagicMock

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
