import os
from time import sleep
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.tool_usage_events import ToolExecutionErrorEvent
from crewai.utilities.token_counter_callback import TokenCalcHandler


# TODO: This test fails without print statement, which makes me think that something is happening asynchronously that we need to eventually fix and dive deeper into at a later date
@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_callback_replacement():
    llm1 = LLM(model="gpt-4o-mini")
    llm2 = LLM(model="gpt-4o-mini")

    calc_handler_1 = TokenCalcHandler(token_cost_process=TokenProcess())
    calc_handler_2 = TokenCalcHandler(token_cost_process=TokenProcess())

    result1 = llm1.call(
        messages=[{"role": "user", "content": "Hello, world!"}],
        callbacks=[calc_handler_1],
    )
    print("result1:", result1)
    usage_metrics_1 = calc_handler_1.token_cost_process.get_summary()
    print("usage_metrics_1:", usage_metrics_1)

    result2 = llm2.call(
        messages=[{"role": "user", "content": "Hello, world from another agent!"}],
        callbacks=[calc_handler_2],
    )
    sleep(5)
    print("result2:", result2)
    usage_metrics_2 = calc_handler_2.token_cost_process.get_summary()
    print("usage_metrics_2:", usage_metrics_2)

    # The first handler should not have been updated
    assert usage_metrics_1.successful_requests == 1
    assert usage_metrics_2.successful_requests == 1
    assert usage_metrics_1 == calc_handler_1.token_cost_process.get_summary()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_string_input():
    llm = LLM(model="gpt-4o-mini")

    # Test the call method with a string input
    result = llm.call("Return the name of a random city in the world.")
    assert isinstance(result, str)
    assert len(result.strip()) > 0  # Ensure the response is not empty


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_string_input_and_callbacks():
    llm = LLM(model="gpt-4o-mini")
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_message_list():
    llm = LLM(model="gpt-4o-mini")
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    # Test the call method with a list of messages
    result = llm.call(messages)
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_tool_and_message_list():
    llm = LLM(model="gpt-4o-mini")

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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_passes_additional_params():
    llm = LLM(
        model="gpt-4o-mini",
        vertex_credentials="test_credentials",
        vertex_project="test_project",
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
    llm = LLM(model="gemini/gemini-1.5-pro")
    assert llm._get_custom_llm_provider() == "gemini"


def test_get_custom_llm_provider_openai():
    llm = LLM(model="gpt-4")
    assert llm._get_custom_llm_provider() == "openai"


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
        llm = LLM(model="gemini/gemini-1.5-pro", response_format=DummyResponse)
        with pytest.raises(ValueError) as excinfo:
            llm._validate_call_params()
        assert "does not support response_format" in str(excinfo.value)


def test_validate_call_params_no_response_format():
    # When no response_format is provided, no validation error should occur.
    llm = LLM(model="gemini/gemini-1.5-pro", response_format=None)
    llm._validate_call_params()


def test_validate_call_params_azure_openai_supported():
    """Test that Azure OpenAI models that should support JSON mode can use response_format."""
    # Test with gpt-4o
    llm = LLM(model="azure/gpt-4o", response_format={"type": "json_object"})
    # Should not raise any error
    llm._validate_call_params()
    
    # Test with gpt-35-turbo
    llm = LLM(model="azure/gpt-35-turbo", response_format={"type": "json_object"})
    # Should not raise any error
    llm._validate_call_params()
    
    # Test with gpt-4-turbo
    llm = LLM(model="azure/gpt-4-turbo", response_format={"type": "json_object"})
    # Should not raise any error
    llm._validate_call_params()


def test_validate_call_params_azure_openai_unsupported():
    """Test that Azure OpenAI models that don't support JSON mode cannot use response_format."""
    # Test with a model that doesn't support JSON mode
    with patch("crewai.llm.supports_response_schema", return_value=False):
        llm = LLM(model="azure/text-davinci-003", response_format={"type": "json_object"})
        with pytest.raises(ValueError) as excinfo:
            llm._validate_call_params()
        assert "does not support response_format" in str(excinfo.value)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_o3_mini_reasoning_effort_high():
    llm = LLM(
        model="o3-mini",
        reasoning_effort="high",
    )
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr(filter_headers=["authorization"])
def test_o3_mini_reasoning_effort_low():
    llm = LLM(
        model="o3-mini",
        reasoning_effort="low",
    )
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
@pytest.fixture
def anthropic_llm():
    """Fixture providing an Anthropic LLM instance."""
    return LLM(model="anthropic/claude-3-sonnet")


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
    with pytest.raises(TypeError, match="Messages cannot be None"):
        anthropic_llm._format_messages_for_provider(None)

    # Test empty message list
    formatted = anthropic_llm._format_messages_for_provider([])
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "."

    # Test invalid message format
    with pytest.raises(TypeError, match="Invalid message format"):
        anthropic_llm._format_messages_for_provider([{"invalid": "message"}])


def test_anthropic_model_detection():
    """Test Anthropic model detection with various formats."""
    models = [
        ("anthropic/claude-3", True),
        ("claude-instant", True),
        ("claude/v1", True),
        ("gpt-4", False),
        ("", False),
        ("anthropomorphic", False),  # Should not match partial words
    ]

    for model, expected in models:
        llm = LLM(model=model)
        assert llm.is_anthropic == expected, f"Failed for model: {model}"


def test_anthropic_message_formatting(anthropic_llm, system_message, user_message):
    """Test Anthropic message formatting with fixtures."""
    # Test when first message is system
    formatted = anthropic_llm._format_messages_for_provider([system_message])
    assert len(formatted) == 2
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "."
    assert formatted[1] == system_message

    # Test when first message is already user
    formatted = anthropic_llm._format_messages_for_provider([user_message])
    assert len(formatted) == 1
    assert formatted[0] == user_message

    # Test with empty message list
    formatted = anthropic_llm._format_messages_for_provider([])
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "."

    # Test with non-Anthropic model (should not modify messages)
    non_anthropic_llm = LLM(model="gpt-4")
    formatted = non_anthropic_llm._format_messages_for_provider([system_message])
    assert len(formatted) == 1
    assert formatted[0] == system_message


def test_deepseek_r1_with_open_router():
    if not os.getenv("OPEN_ROUTER_API_KEY"):
        pytest.skip("OPEN_ROUTER_API_KEY not set; skipping test.")

    llm = LLM(
        model="openrouter/deepseek/deepseek-r1",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    result = llm.call("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr(filter_headers=["authorization"])
def test_tool_execution_error_event():
    llm = LLM(model="gpt-4o-mini")

    def failing_tool(param: str) -> str:
        """This tool always fails."""
        raise Exception("Tool execution failed!")

    tool_schema = {
        "type": "function",
        "function": {
            "name": "failing_tool",
            "description": "This tool always fails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "A test parameter"}
                },
                "required": ["param"],
            },
        },
    }

    received_events = []

    @crewai_event_bus.on(ToolExecutionErrorEvent)
    def event_handler(source, event):
        received_events.append(event)

    available_functions = {"failing_tool": failing_tool}

    messages = [{"role": "user", "content": "Use the failing tool"}]

    llm.call(
        messages,
        tools=[tool_schema],
        available_functions=available_functions,
    )

    assert len(received_events) == 1
    event = received_events[0]
    assert isinstance(event, ToolExecutionErrorEvent)
    assert event.tool_name == "failing_tool"
    assert event.tool_args == {"param": "test"}
    assert event.tool_class == failing_tool
    assert "Tool execution failed!" in event.error
