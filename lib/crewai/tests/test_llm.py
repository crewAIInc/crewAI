import logging
import os
from time import sleep
from unittest.mock import MagicMock, patch

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.events.event_types import (
    LLMCallCompletedEvent,
    LLMStreamChunkEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.utilities.token_counter_callback import TokenCalcHandler
from pydantic import BaseModel
import pytest


# TODO: This test fails without print statement, which makes me think that something is happening asynchronously that we need to eventually fix and dive deeper into at a later date
@pytest.mark.vcr()
def test_llm_callback_replacement():
    llm1 = LLM(model="gpt-4o-mini", is_litellm=True)
    llm2 = LLM(model="gpt-4o-mini", is_litellm=True)

    calc_handler_1 = TokenCalcHandler(token_cost_process=TokenProcess())
    calc_handler_2 = TokenCalcHandler(token_cost_process=TokenProcess())

    llm1.call(
        messages=[{"role": "user", "content": "Hello, world!"}],
        callbacks=[calc_handler_1],
    )
    usage_metrics_1 = calc_handler_1.token_cost_process.get_summary()

    llm2.call(
        messages=[{"role": "user", "content": "Hello, world from another agent!"}],
        callbacks=[calc_handler_2],
    )
    sleep(5)
    usage_metrics_2 = calc_handler_2.token_cost_process.get_summary()

    # The first handler should not have been updated
    assert usage_metrics_1.successful_requests == 1
    assert usage_metrics_2.successful_requests == 1
    assert usage_metrics_1 == calc_handler_1.token_cost_process.get_summary()


@pytest.mark.vcr()
def test_llm_call_with_string_input():
    llm = LLM(model="gpt-4o-mini")

    # Test the call method with a string input
    result = llm.call("Return the name of a random city in the world.")
    assert isinstance(result, str)
    assert len(result.strip()) > 0  # Ensure the response is not empty


@pytest.mark.vcr()
def test_llm_call_with_string_input_and_callbacks():
    llm = LLM(model="gpt-4o-mini", is_litellm=True)
    calc_handler = TokenCalcHandler(token_cost_process=TokenProcess())

    # Test the call method with a string input and callbacks
    result = llm.call(
        "Tell me a joke.",
        callbacks=[calc_handler],
    )
    usage_metrics = calc_handler.token_cost_process.get_summary()

    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert usage_metrics.successful_requests == 1


@pytest.mark.vcr()
def test_llm_call_with_message_list():
    llm = LLM(model="gpt-4o-mini")
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    # Test the call method with a list of messages
    result = llm.call(messages)
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr()
def test_llm_call_with_tool_and_string_input():
    llm = LLM(model="gpt-4o-mini")

    def get_current_year() -> str:
        """Returns the current year as a string."""
        from datetime import datetime

        return str(datetime.now().year)

    # Create tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_current_year",
            "description": "Returns the current year as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    # Available functions mapping
    available_functions = {"get_current_year": get_current_year}

    # Test the call method with a string input and tool
    result = llm.call(
        "What is the current year?",
        tools=[tool_schema],
        available_functions=available_functions,
    )

    assert isinstance(result, str)
    assert result == get_current_year()


@pytest.mark.vcr()
def test_llm_call_with_tool_and_message_list():
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    def square_number(number: int) -> int:
        """Returns the square of a number."""
        return number * number

    # Create tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": "square_number",
            "description": "Returns the square of a number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "description": "The number to square"}
                },
                "required": ["number"],
            },
        },
    }

    # Available functions mapping
    available_functions = {"square_number": square_number}

    messages = [{"role": "user", "content": "What is the square of 5?"}]

    # Test the call method with messages and tool
    result = llm.call(
        messages,
        tools=[tool_schema],
        available_functions=available_functions,
    )

    assert isinstance(result, int)
    assert result == 25


@pytest.mark.vcr()
def test_llm_passes_additional_params():
    llm = LLM(
        model="gpt-4o-mini",
        vertex_credentials="test_credentials",
        vertex_project="test_project",
        is_litellm=True,
    )

    messages = [{"role": "user", "content": "Hello, world!"}]

    with patch("litellm.completion") as mocked_completion:
        # Create mocks for response structure
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 10,
        }

        # Set up the mocked completion to return the mock response
        mocked_completion.return_value = mock_response

        result = llm.call(messages)

        # Assert that litellm.completion was called once
        mocked_completion.assert_called_once()

        # Retrieve the actual arguments with which litellm.completion was called
        _, kwargs = mocked_completion.call_args

        # Check that the additional_params were passed to litellm.completion
        assert kwargs["vertex_credentials"] == "test_credentials"
        assert kwargs["vertex_project"] == "test_project"

        # Also verify that other expected parameters are present
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"] == messages

        # Check the result from llm.call
        assert result == "Test response"


def test_get_custom_llm_provider_openrouter():
    llm = LLM(model="openrouter/deepseek/deepseek-chat")
    assert llm._get_custom_llm_provider() == "openrouter"


def test_get_custom_llm_provider_gemini():
    llm = LLM(model="gemini/gemini-1.5-pro", is_litellm=True)
    assert llm._get_custom_llm_provider() == "gemini"


def test_get_custom_llm_provider_openai():
    llm = LLM(model="gpt-4", is_litellm=True)
    assert llm._get_custom_llm_provider() is None


def test_validate_call_params_supported():
    class DummyResponse(BaseModel):
        a: int

    # Patch supports_response_schema to simulate a supported model.
    with patch("crewai.llm.supports_response_schema", return_value=True):
        llm = LLM(
            model="openrouter/deepseek/deepseek-chat", response_format=DummyResponse
        )
        # Should not raise any error.
        llm._validate_call_params()


def test_validate_call_params_not_supported():
    class DummyResponse(BaseModel):
        a: int

    # Patch supports_response_schema to simulate an unsupported model.
    with patch("crewai.llm.supports_response_schema", return_value=False):
        llm = LLM(
            model="gemini/gemini-1.5-pro",
            response_format=DummyResponse,
            is_litellm=True,
        )
        with pytest.raises(ValueError) as excinfo:
            llm._validate_call_params()
        assert "does not support response_format" in str(excinfo.value)


def test_validate_call_params_no_response_format():
    # When no response_format is provided, no validation error should occur.
    llm = LLM(model="gemini/gemini-1.5-pro", response_format=None, is_litellm=True)
    llm._validate_call_params()


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-3-pro-preview",
        "gemini/gemini-2.0-flash-thinking-exp-01-21",
        "gemini/gemini-2.0-flash-001",
        "gemini/gemini-2.0-flash-lite-001",
    ],
)
def test_gemini_models(model):
    # Use LiteLLM for VCR compatibility (VCR can intercept HTTP calls but not native SDK calls)
    llm = LLM(model=model, is_litellm=False)
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemma-3-27b-it",
    ],
)
def test_gemma3(model):
    # Use LiteLLM for VCR compatibility (VCR can intercept HTTP calls but not native SDK calls)
    llm = LLM(model=model, is_litellm=True)
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "model", ["gpt-4.1", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"]
)
def test_gpt_4_1(model):
    llm = LLM(model=model)
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr()
def test_o3_mini_reasoning_effort_high():
    llm = LLM(
        model="o3-mini",
        reasoning_effort="high",
    )
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr()
def test_o3_mini_reasoning_effort_low():
    llm = LLM(
        model="o3-mini",
        reasoning_effort="low",
    )
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr()
def test_o3_mini_reasoning_effort_medium():
    llm = LLM(
        model="o3-mini",
        reasoning_effort="medium",
    )
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


def test_context_window_validation():
    """Test that context window validation works correctly."""
    # Test valid window size
    llm = LLM(model="o3-mini")
    assert llm.get_context_window_size() == int(200000 * CONTEXT_WINDOW_USAGE_RATIO)

    # Test invalid window size
    with pytest.raises(ValueError) as excinfo:
        with patch.dict(
            "crewai.llm.LLM_CONTEXT_WINDOW_SIZES",
            {"test-model": 500},  # Below minimum
            clear=True,
        ):
            llm = LLM(model="test-model")
            llm.get_context_window_size()
    assert "must be between 1024 and 2097152" in str(excinfo.value)


@pytest.fixture
def get_weather_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    }


def test_context_window_exceeded_error_handling():
    """Test that litellm.ContextWindowExceededError is converted to LLMContextLengthExceededError."""
    from crewai.utilities.exceptions.context_window_exceeding_exception import (
        LLMContextLengthExceededError,
    )
    from litellm.exceptions import ContextWindowExceededError

    llm = LLM(model="gpt-4", is_litellm=True)

    # Test non-streaming response
    with patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = ContextWindowExceededError(
            "This model's maximum context length is 8192 tokens. However, your messages resulted in 10000 tokens.",
            model="gpt-4",
            llm_provider="openai",
        )

        with pytest.raises(LLMContextLengthExceededError) as excinfo:
            llm.call("This is a test message")

        assert "context length exceeded" in str(excinfo.value).lower()
        assert "8192 tokens" in str(excinfo.value)

    # Test streaming response
    llm = LLM(model="gpt-4", stream=True, is_litellm=True)
    with patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = ContextWindowExceededError(
            "This model's maximum context length is 8192 tokens. However, your messages resulted in 10000 tokens.",
            model="gpt-4",
            llm_provider="openai",
        )

        with pytest.raises(LLMContextLengthExceededError) as excinfo:
            llm.call("This is a test message")

        assert "context length exceeded" in str(excinfo.value).lower()
        assert "8192 tokens" in str(excinfo.value)


@pytest.fixture
def anthropic_llm():
    """Fixture providing an Anthropic LLM instance."""
    return LLM(model="anthropic/claude-3-sonnet", is_litellm=False)


@pytest.fixture
def system_message():
    """Fixture providing a system message."""
    return {"role": "system", "content": "test"}


@pytest.fixture
def user_message():
    """Fixture providing a user message."""
    return {"role": "user", "content": "test"}


def test_anthropic_message_formatting_edge_cases(anthropic_llm):
    """Test edge cases for Anthropic message formatting."""
    # Test None messages
    anthropic_llm = AnthropicCompletion(model="claude-3-sonnet", is_litellm=False)
    with pytest.raises(TypeError):
        anthropic_llm._format_messages_for_anthropic(None)

    # Test empty message list - Anthropic requires first message to be from user
    formatted, system_message = anthropic_llm._format_messages_for_anthropic([])
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"

    # Test invalid message format
    with pytest.raises(ValueError, match="must have 'role' and 'content' keys"):
        anthropic_llm._format_messages_for_anthropic([{"invalid": "message"}])


def test_anthropic_model_detection():
    """Test Anthropic model detection with various formats."""
    models = [
        ("anthropic/claude-3", True),
        ("claude-instant", True),
        ("claude/v1", True),
        ("gpt-4", False),
        ("anthropomorphic", False),  # Should not match partial words
    ]

    for model, expected in models:
        llm = LLM(model=model, is_litellm=True)
        assert llm.is_anthropic == expected, f"Failed for model: {model}"


def test_anthropic_message_formatting(anthropic_llm, system_message, user_message):
    """Test Anthropic message formatting with fixtures."""
    # Test when first message is system

    # Test empty message list - Anthropic requires first message to be from user
    formatted, extracted_system = anthropic_llm._format_messages_for_anthropic([])
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"

    # Test invalid message format
    with pytest.raises(ValueError, match="must have 'role' and 'content' keys"):
        anthropic_llm._format_messages_for_anthropic([{"invalid": "message"}])


def test_deepseek_r1_with_open_router():
    if not os.getenv("OPEN_ROUTER_API_KEY"):
        pytest.skip("OPEN_ROUTER_API_KEY not set; skipping test.")

    llm = LLM(
        model="openrouter/deepseek/deepseek-r1",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        is_litellm=True,
    )
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


def assert_event_count(
    mock_emit,
    expected_completed_tool_call: int = 0,
    expected_stream_chunk: int = 0,
    expected_completed_llm_call: int = 0,
    expected_tool_usage_started: int = 0,
    expected_tool_usage_finished: int = 0,
    expected_tool_usage_error: int = 0,
    expected_final_chunk_result: str = "",
):
    event_count = {
        "completed_tool_call": 0,
        "stream_chunk": 0,
        "completed_llm_call": 0,
        "tool_usage_started": 0,
        "tool_usage_finished": 0,
        "tool_usage_error": 0,
    }
    final_chunk_result = ""
    for _call in mock_emit.call_args_list:
        event = _call[1]["event"]

        if (
            isinstance(event, LLMCallCompletedEvent)
            and event.call_type.value == "tool_call"
        ):
            event_count["completed_tool_call"] += 1
        elif isinstance(event, LLMStreamChunkEvent):
            event_count["stream_chunk"] += 1
            final_chunk_result += event.chunk
        elif (
            isinstance(event, LLMCallCompletedEvent)
            and event.call_type.value == "llm_call"
        ):
            event_count["completed_llm_call"] += 1
        elif isinstance(event, ToolUsageStartedEvent):
            event_count["tool_usage_started"] += 1
        elif isinstance(event, ToolUsageFinishedEvent):
            event_count["tool_usage_finished"] += 1
        elif isinstance(event, ToolUsageErrorEvent):
            event_count["tool_usage_error"] += 1
        else:
            continue

    assert event_count["completed_tool_call"] == expected_completed_tool_call
    assert event_count["stream_chunk"] == expected_stream_chunk
    assert event_count["completed_llm_call"] == expected_completed_llm_call
    assert event_count["tool_usage_started"] == expected_tool_usage_started
    assert event_count["tool_usage_finished"] == expected_tool_usage_finished
    assert event_count["tool_usage_error"] == expected_tool_usage_error
    assert final_chunk_result == expected_final_chunk_result


@pytest.fixture
def mock_emit() -> MagicMock:
    from crewai.events.event_bus import CrewAIEventsBus

    with patch.object(CrewAIEventsBus, "emit") as mock_emit:
        yield mock_emit


@pytest.mark.vcr()
def test_handle_streaming_tool_calls(get_weather_tool_schema, mock_emit):
    llm = LLM(model="openai/gpt-4o", stream=True, is_litellm=True)
    response = llm.call(
        messages=[
            {"role": "user", "content": "What is the weather in New York?"},
        ],
        tools=[get_weather_tool_schema],
        available_functions={
            "get_weather": lambda location: f"The weather in {location} is sunny"
        },
    )
    assert response == "The weather in New York, NY is sunny"

    expected_final_chunk_result = (
        '{"location":"New York, NY"}The weather in New York, NY is sunny'
    )
    assert_event_count(
        mock_emit=mock_emit,
        expected_completed_tool_call=1,
        expected_stream_chunk=10,
        expected_completed_llm_call=1,
        expected_tool_usage_started=1,
        expected_tool_usage_finished=1,
        expected_final_chunk_result=expected_final_chunk_result,
    )


@pytest.mark.vcr()
def test_handle_streaming_tool_calls_with_error(get_weather_tool_schema, mock_emit):
    def get_weather_error(location):
        raise Exception("Error")

    llm = LLM(model="openai/gpt-4o", stream=True, is_litellm=True)
    response = llm.call(
        messages=[
            {"role": "user", "content": "What is the weather in New York?"},
        ],
        tools=[get_weather_tool_schema],
        available_functions={"get_weather": get_weather_error},
    )
    assert response == ""
    expected_final_chunk_result = '{"location":"New York, NY"}'
    assert_event_count(
        mock_emit=mock_emit,
        expected_stream_chunk=9,
        expected_completed_llm_call=1,
        expected_tool_usage_started=1,
        expected_tool_usage_error=1,
        expected_final_chunk_result=expected_final_chunk_result,
    )


@pytest.mark.vcr()
def test_handle_streaming_tool_calls_no_available_functions(
    get_weather_tool_schema, mock_emit
):
    llm = LLM(model="openai/gpt-4o", stream=True, is_litellm=True)
    response = llm.call(
        messages=[
            {"role": "user", "content": "What is the weather in New York?"},
        ],
        tools=[get_weather_tool_schema],
    )
    assert response == ""

    assert_event_count(
        mock_emit=mock_emit,
        expected_stream_chunk=9,
        expected_completed_llm_call=1,
        expected_final_chunk_result='{"location":"New York, NY"}',
    )


@pytest.mark.vcr()
def test_handle_streaming_tool_calls_no_tools(mock_emit):
    llm = LLM(model="openai/gpt-4o", stream=True, is_litellm=True)
    response = llm.call(
        messages=[
            {"role": "user", "content": "What is the weather in New York?"},
        ],
    )
    assert (
        response
        == "I'm unable to provide real-time information or current weather updates. For the latest weather information in New York, I recommend checking a reliable weather website or app, such as the National Weather Service, Weather.com, or a similar service."
    )

    assert_event_count(
        mock_emit=mock_emit,
        expected_stream_chunk=46,
        expected_completed_llm_call=1,
        expected_final_chunk_result=response,
    )


@pytest.mark.vcr()
@pytest.mark.skip(reason="Highly flaky on ci")
def test_llm_call_when_stop_is_unsupported(caplog):
    llm = LLM(model="o1-mini", stop=["stop"], is_litellm=True)
    with caplog.at_level(logging.INFO):
        result = llm.call("What is the capital of France?")
        assert "Retrying LLM call without the unsupported 'stop'" in caplog.text
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr()
@pytest.mark.skip(reason="Highly flaky on ci")
def test_llm_call_when_stop_is_unsupported_when_additional_drop_params_is_provided(
    caplog,
):
    llm = LLM(
        model="o1-mini",
        stop=["stop"],
        additional_drop_params=["another_param"],
    )
    with caplog.at_level(logging.INFO):
        result = llm.call("What is the capital of France?")
        assert "Retrying LLM call without the unsupported 'stop'" in caplog.text
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.fixture
def ollama_llm():
    return LLM(model="ollama/llama3.2:3b", is_litellm=True)


def test_ollama_appends_dummy_user_message_when_last_is_assistant(ollama_llm):
    original_messages = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello!"},
    ]

    formatted = ollama_llm._format_messages_for_provider(original_messages)

    assert len(formatted) == len(original_messages) + 1
    assert formatted[-1]["role"] == "user"
    assert formatted[-1]["content"] == ""


def test_ollama_does_not_modify_when_last_is_user(ollama_llm):
    original_messages = [
        {"role": "user", "content": "Tell me a joke."},
    ]

    formatted = ollama_llm._format_messages_for_provider(original_messages)

    assert formatted == original_messages


def test_native_provider_raises_error_when_supported_but_fails():
    """Test that when a native provider is in SUPPORTED_NATIVE_PROVIDERS but fails to instantiate, we raise the error."""
    with patch("crewai.llm.SUPPORTED_NATIVE_PROVIDERS", ["openai"]):
        with patch("crewai.llm.LLM._get_native_provider") as mock_get_native:
            # Mock that provider exists but throws an error when instantiated
            mock_provider = MagicMock()
            mock_provider.side_effect = ValueError(
                "Native provider initialization failed"
            )
            mock_get_native.return_value = mock_provider

            with pytest.raises(ImportError) as excinfo:
                LLM(model="gpt-4", is_litellm=False)

            assert "Error importing native provider" in str(excinfo.value)
            assert "Native provider initialization failed" in str(excinfo.value)


def test_native_provider_falls_back_to_litellm_when_not_in_supported_list():
    """Test that when a provider is not in SUPPORTED_NATIVE_PROVIDERS, we fall back to LiteLLM."""
    with patch("crewai.llm.SUPPORTED_NATIVE_PROVIDERS", ["openai", "anthropic"]):
        # Using a provider not in the supported list
        llm = LLM(model="groq/llama-3.1-70b-versatile", is_litellm=False)

        # Should fall back to LiteLLM
        assert llm.is_litellm is True
        assert llm.model == "groq/llama-3.1-70b-versatile"


def test_prefixed_models_with_valid_constants_use_native_sdk():
    """Test that models with native provider prefixes use native SDK when model is in constants."""
    # Test openai/ prefix with actual OpenAI model in constants → Native SDK
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        llm = LLM(model="openai/gpt-4o", is_litellm=False)
        assert llm.is_litellm is False
        assert llm.provider == "openai"

    # Test anthropic/ prefix with Claude model in constants → Native SDK
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        llm2 = LLM(model="anthropic/claude-opus-4-0", is_litellm=False)
        assert llm2.is_litellm is False
        assert llm2.provider == "anthropic"

    # Test gemini/ prefix with Gemini model in constants → Native SDK
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        llm3 = LLM(model="gemini/gemini-2.5-pro", is_litellm=False)
        assert llm3.is_litellm is False
        assert llm3.provider == "gemini"


def test_prefixed_models_with_invalid_constants_use_litellm():
    """Test that models with native provider prefixes use LiteLLM when model is NOT in constants and does NOT match patterns."""
    # Test openai/ prefix with non-OpenAI model (not in OPENAI_MODELS) → LiteLLM
    llm = LLM(model="openai/gemini-2.5-flash", is_litellm=False)
    assert llm.is_litellm is True
    assert llm.model == "openai/gemini-2.5-flash"

    # Test openai/ prefix with model that doesn't match patterns (e.g. no gpt- prefix) → LiteLLM
    llm2 = LLM(model="openai/custom-finetune-model", is_litellm=False)
    assert llm2.is_litellm is True
    assert llm2.model == "openai/custom-finetune-model"

    # Test anthropic/ prefix with non-Anthropic model → LiteLLM
    llm3 = LLM(model="anthropic/gpt-4o", is_litellm=False)
    assert llm3.is_litellm is True
    assert llm3.model == "anthropic/gpt-4o"


def test_prefixed_models_with_valid_patterns_use_native_sdk():
    """Test that models matching provider patterns use native SDK even if not in constants."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        llm = LLM(model="openai/gpt-future-6", is_litellm=False)
        assert llm.is_litellm is False
        assert llm.provider == "openai"
        assert llm.model == "gpt-future-6"

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        llm2 = LLM(model="anthropic/claude-future-5", is_litellm=False)
        assert llm2.is_litellm is False
        assert llm2.provider == "anthropic"
        assert llm2.model == "claude-future-5"


def test_prefixed_models_with_non_native_providers_use_litellm():
    """Test that models with non-native provider prefixes always use LiteLLM."""
    # Test groq/ prefix (not a native provider) → LiteLLM
    llm = LLM(model="groq/llama-3.3-70b", is_litellm=False)
    assert llm.is_litellm is True
    assert llm.model == "groq/llama-3.3-70b"

    # Test together/ prefix (not a native provider) → LiteLLM
    llm2 = LLM(model="together/qwen-2.5-72b", is_litellm=False)
    assert llm2.is_litellm is True
    assert llm2.model == "together/qwen-2.5-72b"


def test_unprefixed_models_use_native_sdk():
    """Test that unprefixed models use native SDK when model is in constants."""
    # gpt-4o is in OPENAI_MODELS → Native OpenAI SDK
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        llm = LLM(model="gpt-4o", is_litellm=False)
        assert llm.is_litellm is False
        assert llm.provider == "openai"

    # claude-opus-4-0 is in ANTHROPIC_MODELS → Native Anthropic SDK
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        llm2 = LLM(model="claude-opus-4-0", is_litellm=False)
        assert llm2.is_litellm is False
        assert llm2.provider == "anthropic"

    # gemini-2.5-pro is in GEMINI_MODELS → Native Gemini SDK
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        llm3 = LLM(model="gemini-2.5-pro", is_litellm=False)
        assert llm3.is_litellm is False
        assert llm3.provider == "gemini"


def test_explicit_provider_kwarg_takes_priority():
    """Test that explicit provider kwarg takes priority over model name inference."""
    # Explicit provider=openai should use OpenAI even if model name suggests otherwise
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        llm = LLM(model="gpt-4o", provider="openai", is_litellm=False)
        assert llm.is_litellm is False
        assert llm.provider == "openai"

    # Explicit provider for a model with "/" should still use that provider
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        llm2 = LLM(model="gpt-4o", provider="openai", is_litellm=False)
        assert llm2.is_litellm is False
        assert llm2.provider == "openai"


def test_validate_model_in_constants():
    """Test the _validate_model_in_constants method."""
    # OpenAI models
    assert LLM._validate_model_in_constants("gpt-4o", "openai") is True
    assert LLM._validate_model_in_constants("gpt-future-6", "openai") is True
    assert LLM._validate_model_in_constants("o1-latest", "openai") is True
    assert LLM._validate_model_in_constants("unknown-model", "openai") is False

    # Anthropic models
    assert LLM._validate_model_in_constants("claude-opus-4-0", "claude") is True
    assert LLM._validate_model_in_constants("claude-future-5", "claude") is True
    assert (
        LLM._validate_model_in_constants("claude-3-5-sonnet-latest", "claude") is True
    )
    assert LLM._validate_model_in_constants("unknown-model", "claude") is False

    # Gemini models
    assert LLM._validate_model_in_constants("gemini-2.5-pro", "gemini") is True
    assert LLM._validate_model_in_constants("gemini-future", "gemini") is True
    assert LLM._validate_model_in_constants("gemma-3-latest", "gemini") is True
    assert LLM._validate_model_in_constants("unknown-model", "gemini") is False

    # Azure models
    assert LLM._validate_model_in_constants("gpt-4o", "azure") is True
    assert LLM._validate_model_in_constants("gpt-35-turbo", "azure") is True

    # Bedrock models
    assert (
        LLM._validate_model_in_constants(
            "anthropic.claude-opus-4-1-20250805-v1:0", "bedrock"
        )
        is True
    )
    assert (
        LLM._validate_model_in_constants("anthropic.claude-future-v1:0", "bedrock")
        is True
    )
