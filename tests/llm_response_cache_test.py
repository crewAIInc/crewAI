import pytest
from unittest.mock import MagicMock, patch

from crewai.llm import LLM
from crewai.utilities.llm_response_cache_handler import LLMResponseCacheHandler


@pytest.fixture
def handler():
    handler = LLMResponseCacheHandler()
    handler.storage.add = MagicMock()
    handler.storage.get = MagicMock()
    return handler


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_recording(handler):
    handler.start_recording()
    
    llm = LLM(model="gpt-4o-mini")
    llm.set_response_cache_handler(handler)
    
    messages = [{"role": "user", "content": "Hello, world!"}]
    
    with patch('litellm.completion') as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello, human!"}}]
        }
        
        response = llm.call(messages)
        
        assert response == "Hello, human!"
        
        handler.storage.add.assert_called_once_with(
            "gpt-4o-mini", messages, "Hello, human!"
        )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_replaying(handler):
    handler.start_replaying()
    handler.storage.get.return_value = "Cached response"
    
    llm = LLM(model="gpt-4o-mini")
    llm.set_response_cache_handler(handler)
    
    messages = [{"role": "user", "content": "Hello, world!"}]
    
    with patch('litellm.completion') as mock_completion:
        response = llm.call(messages)
        
        assert response == "Cached response"
        
        mock_completion.assert_not_called()
        
        handler.storage.get.assert_called_once_with("gpt-4o-mini", messages)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_replay_fallback(handler):
    handler.start_replaying()
    handler.storage.get.return_value = None
    
    llm = LLM(model="gpt-4o-mini")
    llm.set_response_cache_handler(handler)
    
    messages = [{"role": "user", "content": "Hello, world!"}]
    
    with patch('litellm.completion') as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello, human!"}}]
        }
        
        response = llm.call(messages)
        
        assert response == "Hello, human!"
        
        mock_completion.assert_called_once()
