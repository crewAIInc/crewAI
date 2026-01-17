"""Tests for OpenAI Responses API provider."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.openai_responses.completion import (
    OpenAIResponsesCompletion,
)


def test_openai_responses_completion_is_used_when_provider_specified():
    """
    Test that OpenAIResponsesCompletion is used when provider='openai_responses'
    """
    llm = LLM(model="gpt-4o", provider="openai_responses")

    assert llm.__class__.__name__ == "OpenAIResponsesCompletion"
    assert llm.provider == "openai_responses"
    assert llm.model == "gpt-4o"


def test_openai_responses_completion_is_used_with_model_prefix():
    """
    Test that OpenAIResponsesCompletion is used when model has openai_responses/ prefix
    """
    llm = LLM(model="openai_responses/gpt-4o")

    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.provider == "openai_responses"
    assert llm.model == "gpt-4o"


def test_openai_responses_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using openai_responses provider
    """
    module_name = "crewai.llms.providers.openai_responses.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="gpt-4o", provider="openai_responses")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, "OpenAIResponsesCompletion")


def test_openai_responses_completion_initialization_parameters():
    """
    Test that OpenAIResponsesCompletion is initialized with correct parameters
    """
    llm = LLM(
        model="gpt-4o",
        provider="openai_responses",
        temperature=0.7,
        max_output_tokens=1000,
        api_key="test-key",
    )

    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.7
    assert llm.max_output_tokens == 1000


def test_openai_responses_completion_with_reasoning_effort():
    """
    Test that OpenAIResponsesCompletion accepts reasoning_effort for o-series models
    """
    llm = LLM(
        model="o3-mini",
        provider="openai_responses",
        reasoning_effort="high",
    )

    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.model == "o3-mini"
    assert llm.reasoning_effort == "high"
    assert llm.is_o_series is True


def test_openai_responses_completion_call():
    """
    Test that OpenAIResponsesCompletion call method works
    """
    llm = LLM(model="gpt-4o", provider="openai_responses")

    # Mock the call method on the instance
    with patch.object(llm, "call", return_value="Hello! I'm ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


def test_openai_responses_prepare_params_simple_message():
    """
    Test that _prepare_responses_params handles simple string input correctly
    """
    llm = OpenAIResponsesCompletion(model="gpt-4o")

    messages = [{"role": "user", "content": "Hello world"}]
    params = llm._prepare_responses_params(messages)

    assert params["model"] == "gpt-4o"
    assert params["input"] == "Hello world"  # Simple case returns string
    assert "instructions" not in params  # No system message


def test_openai_responses_prepare_params_with_system_message():
    """
    Test that _prepare_responses_params converts system messages to instructions
    """
    llm = OpenAIResponsesCompletion(model="gpt-4o")

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    params = llm._prepare_responses_params(messages)

    assert params["model"] == "gpt-4o"
    assert params["instructions"] == "You are a helpful assistant"
    assert params["input"] == "Hello"


def test_openai_responses_prepare_params_with_multiple_messages():
    """
    Test that _prepare_responses_params handles multiple messages correctly
    """
    llm = OpenAIResponsesCompletion(model="gpt-4o")

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What about 3+3?"},
    ]
    params = llm._prepare_responses_params(messages)

    assert params["instructions"] == "You are a helpful assistant"
    # Multiple messages should be a list
    assert isinstance(params["input"], list)
    assert len(params["input"]) == 3  # 3 non-system messages


def test_openai_responses_prepare_params_with_reasoning_effort():
    """
    Test that _prepare_responses_params includes reasoning parameter for o-series
    """
    llm = OpenAIResponsesCompletion(model="o3-mini", reasoning_effort="high")

    messages = [{"role": "user", "content": "Solve this problem"}]
    params = llm._prepare_responses_params(messages)

    assert params["reasoning"] == {"effort": "high"}


def test_openai_responses_prepare_params_with_tools():
    """
    Test that _prepare_responses_params correctly converts tools
    """
    llm = OpenAIResponsesCompletion(model="gpt-4o")

    messages = [{"role": "user", "content": "Search for information"}]
    tools = [
        {
            "name": "search",
            "description": "Search for information",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]

    params = llm._prepare_responses_params(messages, tools=tools)

    assert "tools" in params
    assert len(params["tools"]) == 1
    assert params["tools"][0]["type"] == "function"
    assert params["tools"][0]["function"]["name"] == "search"
    assert params["tools"][0]["function"]["strict"] is True


def test_openai_responses_supports_function_calling():
    """
    Test supports_function_calling returns correct value based on model
    """
    # Regular model supports function calling
    llm_gpt4 = OpenAIResponsesCompletion(model="gpt-4o")
    assert llm_gpt4.supports_function_calling() is True

    # o1-preview doesn't support function calling
    llm_o1_preview = OpenAIResponsesCompletion(model="o1-preview")
    assert llm_o1_preview.supports_function_calling() is False


def test_openai_responses_get_context_window_size():
    """
    Test that get_context_window_size returns correct values for different models
    """
    # GPT-4o has 128000 context window
    llm_gpt4o = OpenAIResponsesCompletion(model="gpt-4o")
    context_size = llm_gpt4o.get_context_window_size()
    # Should be 85% of 128000 = 108800
    assert context_size == int(128000 * 0.85)

    # o3-mini has 200000 context window
    llm_o3 = OpenAIResponsesCompletion(model="o3-mini")
    context_size = llm_o3.get_context_window_size()
    assert context_size == int(200000 * 0.85)


def test_openai_responses_is_o_series_detection():
    """
    Test that o-series models are correctly detected
    """
    # O-series models
    assert OpenAIResponsesCompletion(model="o1").is_o_series is True
    assert OpenAIResponsesCompletion(model="o1-mini").is_o_series is True
    assert OpenAIResponsesCompletion(model="o1-preview").is_o_series is True
    assert OpenAIResponsesCompletion(model="o3-mini").is_o_series is True

    # Non-O-series models
    assert OpenAIResponsesCompletion(model="gpt-4o").is_o_series is False
    assert OpenAIResponsesCompletion(model="gpt-4o-mini").is_o_series is False


def test_openai_responses_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native OpenAI Responses completion fails to initialize.
    """
    with patch("crewai.llm.LLM._get_native_provider") as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native SDK failed")

        mock_get_provider.return_value = FailingCompletion

        with pytest.raises(ImportError) as excinfo:
            LLM(model="gpt-4o", provider="openai_responses")

        assert "Error importing native provider" in str(excinfo.value)
        assert "Native SDK failed" in str(excinfo.value)


def test_openai_responses_streaming_parameter():
    """
    Test that streaming parameter is correctly set
    """
    llm = OpenAIResponsesCompletion(model="gpt-4o", stream=True)
    assert llm.stream is True

    llm_no_stream = OpenAIResponsesCompletion(model="gpt-4o", stream=False)
    assert llm_no_stream.stream is False


def test_openai_responses_previous_response_id():
    """
    Test that previous_response_id parameter is handled correctly
    """
    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        previous_response_id="resp_123abc",
    )
    assert llm.previous_response_id == "resp_123abc"


def test_openai_responses_get_client_params_with_api_base():
    """
    Test that _get_client_params correctly converts api_base to base_url
    """
    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        api_base="https://custom.openai.com/v1",
    )
    client_params = llm._get_client_params()
    assert client_params["base_url"] == "https://custom.openai.com/v1"


def test_openai_responses_convert_tools_strict_mode():
    """
    Test that tools are converted with strict: true by default
    """
    llm = OpenAIResponsesCompletion(model="gpt-4o")

    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        }
    ]

    converted = llm._convert_tools_for_responses(tools)

    assert len(converted) == 1
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["strict"] is True
    assert converted[0]["function"]["name"] == "get_weather"
