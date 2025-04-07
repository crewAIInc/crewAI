import pytest
from unittest.mock import patch, Mock
from contextlib import nullcontext as does_not_raise

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


def test_cache_ttl_validation():
    """Test that cache_ttl validation works correctly."""
    with does_not_raise():
        LLM(model="gpt-4o-mini", cache_enabled=True, cache_ttl=3600)
    
    with pytest.raises(ValueError, match="cache_ttl must be positive"):
        LLM(model="gpt-4o-mini", cache_enabled=True, cache_ttl=-1)
    
    with pytest.raises(ValueError, match="cache_ttl must be positive"):
        LLM(model="gpt-4o-mini", cache_enabled=True, cache_ttl=0)


def test_cache_disable_runtime():
    """Test that cache can be disabled at runtime."""
    llm = LLM(model="gpt-4o-mini", cache_enabled=True)
    assert llm.cache_enabled is True
    
    llm.cache_enabled = False
    assert llm.cache_enabled is False
    
    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_completion.return_value = mock_response.return_value
        
        llm.call(messages=[{"role": "user", "content": "Hello, world!"}])
        
        call_args = mock_completion.call_args[1]
        assert "cache_control" not in call_args


def test_temporary_cache_settings():
    """Test that temporary_cache_settings context manager works correctly."""
    llm = LLM(model="gpt-4o-mini", cache_enabled=False)
    assert llm.cache_enabled is False
    assert llm.cache_ttl is None
    
    with llm.temporary_cache_settings(enabled=True, ttl=3600):
        assert llm.cache_enabled is True
        assert llm.cache_ttl == 3600
    
    assert llm.cache_enabled is False
    assert llm.cache_ttl is None


@pytest.mark.parametrize("model", ["gpt-4o", "gemini-1.5-pro"])
def test_cache_provider_specific(model):
    """Test cache behavior across different models."""
    llm = LLM(model=model, cache_enabled=True, cache_ttl=3600)
    
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
