import pytest
from unittest.mock import patch, MagicMock

from crewai.llm import LLM
from crewai.agent import Agent


@pytest.mark.vcr(filter_headers=["authorization"])
@patch("litellm.Router")
@patch.object(LLM, '_initialize_router')
def test_llm_with_model_list(mock_initialize_router, mock_router):
    """Test that LLM can be initialized with a model_list for multiple model configurations."""
    mock_initialize_router.return_value = None
    
    mock_router_instance = MagicMock()
    mock_router.return_value = mock_router_instance
    
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": "test-key-1"
            }
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key-2"
            }
        }
    ]
    
    llm = LLM(model="gpt-4o-mini", model_list=model_list)
    llm.router = mock_router_instance
    
    assert llm.model == "gpt-4o-mini"
    assert llm.model_list == model_list
    assert llm.router is not None


@pytest.mark.vcr(filter_headers=["authorization"])
@patch("litellm.Router")
@patch.object(LLM, '_initialize_router')
def test_llm_with_routing_strategy(mock_initialize_router, mock_router):
    """Test that LLM can be initialized with a routing strategy."""
    mock_initialize_router.return_value = None
    
    mock_router_instance = MagicMock()
    mock_router.return_value = mock_router_instance
    
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": "test-key-1"
            }
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key-2"
            }
        }
    ]
    
    llm = LLM(
        model="gpt-4o-mini", 
        model_list=model_list, 
        routing_strategy="simple-shuffle"
    )
    llm.router = mock_router_instance
    
    assert llm.routing_strategy == "simple-shuffle"
    assert llm.router is not None


@pytest.mark.vcr(filter_headers=["authorization"])
@patch("litellm.Router")
@patch.object(LLM, '_initialize_router')
def test_agent_with_model_list(mock_initialize_router, mock_router):
    """Test that Agent can be initialized with a model_list for multiple model configurations."""
    mock_initialize_router.return_value = None
    
    mock_router_instance = MagicMock()
    mock_router.return_value = mock_router_instance
    
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": "test-key-1"
            }
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key-2"
            }
        }
    ]
    
    with patch.object(Agent, 'post_init_setup', wraps=Agent.post_init_setup) as mock_post_init:
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            model_list=model_list
        )
        
        agent.llm.router = mock_router_instance
        
        assert agent.model_list == model_list
        assert agent.llm.model_list == model_list
        assert agent.llm.router is not None


@pytest.mark.vcr(filter_headers=["authorization"])
@patch("litellm.Router")
@patch.object(LLM, '_initialize_router')
def test_llm_call_with_router(mock_initialize_router, mock_router):
    """Test that LLM.call uses the router when model_list is provided."""
    mock_initialize_router.return_value = None
    
    mock_router_instance = MagicMock()
    mock_router.return_value = mock_router_instance
    
    mock_response = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_router_instance.completion.return_value = mock_response
    
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": "test-key-1"
            }
        }
    ]
    
    # Create LLM with model_list
    llm = LLM(model="gpt-4o-mini", model_list=model_list)
    
    llm.router = mock_router_instance
    
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.call(messages)
    
    mock_router_instance.completion.assert_called_once()
    assert response == "Test response"


@pytest.mark.vcr(filter_headers=["authorization"])
@patch("litellm.completion")
def test_llm_call_without_router(mock_completion):
    """Test that LLM.call uses litellm.completion when no model_list is provided."""
    mock_response = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_completion.return_value = mock_response
    
    llm = LLM(model="gpt-4o-mini")
    
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.call(messages)
    
    mock_completion.assert_called_once()
    assert response == "Test response"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_with_invalid_routing_strategy():
    """Test that LLM initialization raises an error with an invalid routing strategy."""
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": "test-key-1"
            }
        }
    ]
    
    with pytest.raises(RuntimeError) as exc_info:
        LLM(
            model="gpt-4o-mini", 
            model_list=model_list, 
            routing_strategy="invalid-strategy"
        )
    
    assert "Invalid routing strategy" in str(exc_info.value)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_with_invalid_routing_strategy():
    """Test that Agent initialization raises an error with an invalid routing strategy."""
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": "test-key-1"
            }
        }
    ]
    
    with pytest.raises(Exception) as exc_info:
        Agent(
            role="test",
            goal="test",
            backstory="test",
            model_list=model_list,
            routing_strategy="invalid-strategy"
        )
    
    assert "Input should be" in str(exc_info.value)
    assert "simple-shuffle" in str(exc_info.value)
    assert "least-busy" in str(exc_info.value)


@pytest.mark.vcr(filter_headers=["authorization"])
@patch.object(LLM, '_initialize_router')
def test_llm_with_missing_model_in_litellm_params(mock_initialize_router):
    """Test that LLM initialization raises an error when model is missing in litellm_params."""
    mock_initialize_router.side_effect = RuntimeError("Router initialization failed: Missing required 'model' in litellm_params")
    
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "api_key": "test-key-1"
            }
        }
    ]
    
    with pytest.raises(RuntimeError) as exc_info:
        LLM(model="gpt-4o-mini", model_list=model_list)
    
    assert "Router initialization failed" in str(exc_info.value)
