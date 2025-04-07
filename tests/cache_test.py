import pytest
from unittest.mock import patch, Mock

from crewai.llm import LLM


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_cache_control():
    """Test that cache_control is correctly passed to litellm when enabled."""
    llm = LLM(model="gpt-4o-mini", cache_enabled=True)
    
    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_completion.return_value = mock_response.return_value
        
        llm.call(messages=[{"role": "user", "content": "Hello, world!"}])
        
        call_args = mock_completion.call_args[1]
        assert "cache_control" in call_args
        assert call_args["cache_control"]["enabled"] is True


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_cache_ttl():
    """Test that cache_ttl is correctly passed to litellm when specified."""
    llm = LLM(model="gpt-4o-mini", cache_enabled=True, cache_ttl=3600)
    
    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_completion.return_value = mock_response.return_value
        
        llm.call(messages=[{"role": "user", "content": "Hello, world!"}])
        
        call_args = mock_completion.call_args[1]
        assert "cache_control" in call_args
        assert call_args["cache_control"]["enabled"] is True
        assert call_args["cache_control"]["ttl"] == 3600
