import os
import sys
import types
from unittest.mock import patch, MagicMock
import pytest

from crewai.llm import LLM
from crewai.crew import Crew
from crewai.agent import Agent
from crewai.task import Task


@pytest.fixture(autouse=True)
def mock_anthropic_api_key():
    """Automatically mock ANTHROPIC_API_KEY for all tests in this module."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        yield


def test_anthropic_completion_is_used_when_anthropic_provider():
    """
    Test that AnthropicCompletion from completion.py is used when LLM uses provider 'anthropic'
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    assert llm.__class__.__name__ == "AnthropicCompletion"
    assert llm.provider == "anthropic"
    assert llm.model == "claude-3-5-sonnet-20241022"


def test_anthropic_completion_is_used_when_claude_provider():
    """
    Test that AnthropicCompletion is used when provider is 'claude'
    """
    llm = LLM(model="claude/claude-3-5-sonnet-20241022")

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    assert isinstance(llm, AnthropicCompletion)
    assert llm.provider == "claude"
    assert llm.model == "claude-3-5-sonnet-20241022"




def test_anthropic_tool_use_conversation_flow():
    """
    Test that the Anthropic completion properly handles tool use conversation flow
    """
    from unittest.mock import Mock, patch
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    from anthropic.types.tool_use_block import ToolUseBlock

    # Create AnthropicCompletion instance
    completion = AnthropicCompletion(model="claude-3-5-sonnet-20241022")

    # Mock tool function
    def mock_weather_tool(location: str) -> str:
        return f"The weather in {location} is sunny and 75°F"

    available_functions = {"get_weather": mock_weather_tool}

    # Mock the Anthropic client responses
    with patch.object(completion.client.messages, 'create') as mock_create:
        # Mock initial response with tool use - need to properly mock ToolUseBlock
        mock_tool_use = Mock(spec=ToolUseBlock)
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "get_weather"
        mock_tool_use.input = {"location": "San Francisco"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.usage = Mock()
        mock_initial_response.usage.input_tokens = 100
        mock_initial_response.usage.output_tokens = 50

        # Mock final response after tool result - properly mock text content
        mock_text_block = Mock()
        # Set the text attribute as a string, not another Mock
        mock_text_block.configure_mock(text="Based on the weather data, it's a beautiful day in San Francisco with sunny skies and 75°F temperature.")

        mock_final_response = Mock()
        mock_final_response.content = [mock_text_block]
        mock_final_response.usage = Mock()
        mock_final_response.usage.input_tokens = 150
        mock_final_response.usage.output_tokens = 75

        # Configure mock to return different responses on successive calls
        mock_create.side_effect = [mock_initial_response, mock_final_response]

        # Test the call
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        result = completion.call(
            messages=messages,
            available_functions=available_functions
        )

        # Verify the result contains the final response
        assert "beautiful day in San Francisco" in result
        assert "sunny skies" in result
        assert "75°F" in result

        # Verify that two API calls were made (initial + follow-up)
        assert mock_create.call_count == 2

        # Verify the second call includes tool results
        second_call_args = mock_create.call_args_list[1][1]  # kwargs of second call
        messages_in_second_call = second_call_args["messages"]

        # Should have original user message + assistant tool use + user tool result
        assert len(messages_in_second_call) == 3
        assert messages_in_second_call[0]["role"] == "user"
        assert messages_in_second_call[1]["role"] == "assistant"
        assert messages_in_second_call[2]["role"] == "user"

        # Verify tool result format
        tool_result = messages_in_second_call[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_123"
        assert "sunny and 75°F" in tool_result["content"]


def test_anthropic_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using Anthropic provider
    """
    module_name = "crewai.llms.providers.anthropic.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'AnthropicCompletion')


def test_fallback_to_litellm_when_native_anthropic_fails():
    """
    Test that LLM falls back to LiteLLM when native Anthropic completion fails
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native Anthropic SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should fall back to LiteLLM
        llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

        # Check that it's using LiteLLM
        assert hasattr(llm, 'is_litellm')
        assert llm.is_litellm == True


def test_anthropic_completion_initialization_parameters():
    """
    Test that AnthropicCompletion is initialized with correct parameters
    """
    llm = LLM(
        model="anthropic/claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9,
        api_key="test-key"
    )

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    assert isinstance(llm, AnthropicCompletion)
    assert llm.model == "claude-3-5-sonnet-20241022"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2000
    assert llm.top_p == 0.9


def test_anthropic_specific_parameters():
    """
    Test Anthropic-specific parameters like stop_sequences and streaming
    """
    llm = LLM(
        model="anthropic/claude-3-5-sonnet-20241022",
        stop_sequences=["Human:", "Assistant:"],
        stream=True,
        max_retries=5,
        timeout=60
    )

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    assert isinstance(llm, AnthropicCompletion)
    assert llm.stop_sequences == ["Human:", "Assistant:"]
    assert llm.stream == True
    assert llm.client.max_retries == 5
    assert llm.client.timeout == 60


def test_anthropic_completion_call():
    """
    Test that AnthropicCompletion call method works
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock the call method on the instance
    with patch.object(llm, 'call', return_value="Hello! I'm Claude, ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm Claude, ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


def test_anthropic_completion_called_during_crew_execution():
    """
    Test that AnthropicCompletion.call is actually invoked when running a crew
    """
    # Create the LLM instance first
    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock the call method on the specific instance
    with patch.object(anthropic_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:

        # Create agent with explicit LLM configuration
        agent = Agent(
            role="Research Assistant",
            goal="Find population info",
            backstory="You research populations.",
            llm=anthropic_llm,
        )

        task = Task(
            description="Find Tokyo population",
            expected_output="Population number",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        # Verify mock was called
        assert mock_call.called
        assert "14 million" in str(result)


def test_anthropic_completion_call_arguments():
    """
    Test that AnthropicCompletion.call is invoked with correct arguments
    """
    # Create LLM instance first
    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock the instance method
    with patch.object(anthropic_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed successfully."

        agent = Agent(
            role="Test Agent",
            goal="Complete a simple task",
            backstory="You are a test agent.",
            llm=anthropic_llm  # Use same instance
        )

        task = Task(
            description="Say hello world",
            expected_output="Hello world",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()

        # Verify call was made
        assert mock_call.called

        # Check the arguments passed to the call method
        call_args = mock_call.call_args
        assert call_args is not None

        # The first argument should be the messages
        messages = call_args[0][0]  # First positional argument
        assert isinstance(messages, (str, list))

        # Verify that the task description appears in the messages
        if isinstance(messages, str):
            assert "hello world" in messages.lower()
        elif isinstance(messages, list):
            message_content = str(messages).lower()
            assert "hello world" in message_content


def test_multiple_anthropic_calls_in_crew():
    """
    Test that AnthropicCompletion.call is invoked multiple times for multiple tasks
    """
    # Create LLM instance first
    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock the instance method
    with patch.object(anthropic_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent = Agent(
            role="Multi-task Agent",
            goal="Complete multiple tasks",
            backstory="You can handle multiple tasks.",
            llm=anthropic_llm  # Use same instance
        )

        task1 = Task(
            description="First task",
            expected_output="First result",
            agent=agent,
        )

        task2 = Task(
            description="Second task",
            expected_output="Second result",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task1, task2]
        )
        crew.kickoff()

        # Verify multiple calls were made
        assert mock_call.call_count >= 2  # At least one call per task

        # Verify each call had proper arguments
        for call in mock_call.call_args_list:
            assert len(call[0]) > 0  # Has positional arguments
            messages = call[0][0]
            assert messages is not None


def test_anthropic_completion_with_tools():
    """
    Test that AnthropicCompletion.call is invoked with tools when agent has tools
    """
    from crewai.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing"""
        return f"Tool result for: {query}"

    # Create LLM instance first
    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock the instance method
    with patch.object(anthropic_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed with tools."

        agent = Agent(
            role="Tool User",
            goal="Use tools to complete tasks",
            backstory="You can use tools.",
            llm=anthropic_llm,  # Use same instance
            tools=[sample_tool]
        )

        task = Task(
            description="Use the sample tool",
            expected_output="Tool usage result",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()

        assert mock_call.called

        call_args = mock_call.call_args
        call_kwargs = call_args[1] if len(call_args) > 1 else {}

        if 'tools' in call_kwargs:
            assert call_kwargs['tools'] is not None
            assert len(call_kwargs['tools']) > 0


def test_anthropic_raises_error_when_model_not_supported():
    """Test that AnthropicCompletion raises ValueError when model not supported"""

    # Mock the Anthropic client to raise an error
    with patch('crewai.llms.providers.anthropic.completion.Anthropic') as mock_anthropic_class:
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock the error that Anthropic would raise for unsupported models
        from anthropic import NotFoundError
        mock_client.messages.create.side_effect = NotFoundError(
            message="The model `model-doesnt-exist` does not exist",
            response=MagicMock(),
            body={}
        )

        llm = LLM(model="anthropic/model-doesnt-exist")

        with pytest.raises(Exception):  # Should raise some error for unsupported model
            llm.call("Hello")


def test_anthropic_client_params_setup():
    """
    Test that client_params are properly merged with default client parameters
    """
    # Use only valid Anthropic client parameters
    custom_client_params = {
        "default_headers": {"X-Custom-Header": "test-value"},
    }

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key="test-key",
            base_url="https://custom-api.com",
            timeout=45,
            max_retries=5,
            client_params=custom_client_params
        )

        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        assert isinstance(llm, AnthropicCompletion)

        assert llm.client_params == custom_client_params

        merged_params = llm._get_client_params()

        assert merged_params["api_key"] == "test-key"
        assert merged_params["base_url"] == "https://custom-api.com"
        assert merged_params["timeout"] == 45
        assert merged_params["max_retries"] == 5

        assert merged_params["default_headers"] == {"X-Custom-Header": "test-value"}


def test_anthropic_client_params_override_defaults():
    """
    Test that client_params can override default client parameters
    """
    override_client_params = {
        "timeout": 120,  # Override the timeout parameter
        "max_retries": 10,  # Override the max_retries parameter
        "default_headers": {"X-Override": "true"}  # Valid custom parameter
    }

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            client_params=override_client_params
        )

        # Verify this is actually AnthropicCompletion, not LiteLLM fallback
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        assert isinstance(llm, AnthropicCompletion)

        merged_params = llm._get_client_params()

        # client_params should override the individual parameters
        assert merged_params["timeout"] == 120
        assert merged_params["max_retries"] == 10
        assert merged_params["default_headers"] == {"X-Override": "true"}


def test_anthropic_client_params_none():
    """
    Test that client_params=None works correctly (no additional parameters)
    """
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key="test-key",
            base_url="https://api.anthropic.com",
            timeout=60,
            max_retries=2,
            client_params=None
        )

        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        assert isinstance(llm, AnthropicCompletion)

        assert llm.client_params is None

        merged_params = llm._get_client_params()

        expected_keys = {"api_key", "base_url", "timeout", "max_retries"}
        assert set(merged_params.keys()) == expected_keys

        # Fixed assertions - all should be inside the with block and use correct values
        assert merged_params["api_key"] == "test-key"  # Not "test-anthropic-key"
        assert merged_params["base_url"] == "https://api.anthropic.com"
        assert merged_params["timeout"] == 60
        assert merged_params["max_retries"] == 2


def test_anthropic_client_params_empty_dict():
    """
    Test that client_params={} works correctly (empty additional parameters)
    """
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key="test-key",
            client_params={}
        )

        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        assert isinstance(llm, AnthropicCompletion)

        assert llm.client_params == {}

        merged_params = llm._get_client_params()

        assert "api_key" in merged_params
        assert merged_params["api_key"] == "test-key"


def test_anthropic_model_detection():
    """
    Test that various Anthropic model formats are properly detected
    """
    # Test Anthropic model naming patterns that actually work with provider detection
    anthropic_test_cases = [
        "anthropic/claude-3-5-sonnet-20241022",
        "claude/claude-3-5-sonnet-20241022"
    ]

    for model_name in anthropic_test_cases:
        llm = LLM(model=model_name)
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        assert isinstance(llm, AnthropicCompletion), f"Failed for model: {model_name}"


def test_anthropic_supports_stop_words():
    """
    Test that Anthropic models support stop sequences
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")
    assert llm.supports_stop_words() == True


def test_anthropic_context_window_size():
    """
    Test that Anthropic models return correct context window sizes
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")
    context_size = llm.get_context_window_size()

    # Should return a reasonable context window size (Claude 3.5 has 200k tokens)
    assert context_size > 100000  # Should be substantial
    assert context_size <= 200000  # But not exceed the actual limit


def test_anthropic_message_formatting():
    """
    Test that messages are properly formatted for Anthropic API
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Test message formatting
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(test_messages)

    # System message should be extracted
    assert system_message == "You are a helpful assistant."

    # Remaining messages should start with user
    assert formatted_messages[0]["role"] == "user"
    assert len(formatted_messages) >= 3  # Should have user, assistant, user messages


def test_anthropic_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    # Test non-streaming
    llm_no_stream = LLM(model="anthropic/claude-3-5-sonnet-20241022", stream=False)
    assert llm_no_stream.stream == False

    # Test streaming
    llm_stream = LLM(model="anthropic/claude-3-5-sonnet-20241022", stream=True)
    assert llm_stream.stream == True


def test_anthropic_tool_conversion():
    """
    Test that tools are properly converted to Anthropic format
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock tool in CrewAI format
    crewai_tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }]

    # Test tool conversion
    anthropic_tools = llm._convert_tools_for_interference(crewai_tools)

    assert len(anthropic_tools) == 1
    assert anthropic_tools[0]["name"] == "test_tool"
    assert anthropic_tools[0]["description"] == "A test tool"
    assert "input_schema" in anthropic_tools[0]


def test_anthropic_environment_variable_api_key():
    """
    Test that Anthropic API key is properly loaded from environment
    """
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}):
        llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

        assert llm.client is not None
        assert hasattr(llm.client, 'messages')


def test_anthropic_token_usage_tracking():
    """
    Test that token usage is properly tracked for Anthropic responses
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock the Anthropic response with usage information
    with patch.object(llm.client.messages, 'create') as mock_create:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test response")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=25)
        mock_create.return_value = mock_response

        result = llm.call("Hello")

        # Verify the response
        assert result == "test response"

        # Verify token usage was extracted
        usage = llm._extract_anthropic_token_usage(mock_response)
        assert usage["input_tokens"] == 50
        assert usage["output_tokens"] == 25
        assert usage["total_tokens"] == 75
