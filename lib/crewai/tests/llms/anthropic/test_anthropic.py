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
    """Automatically mock ANTHROPIC_API_KEY for all tests in this module if not already set."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            yield
    else:
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
    assert llm.provider == "anthropic"
    assert llm.model == "claude-3-5-sonnet-20241022"




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


def test_native_anthropic_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native Anthropic completion fails to initialize.
    This ensures we don't silently fall back when there's a configuration issue.
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native Anthropic SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should raise ImportError, not fall back to LiteLLM
        with pytest.raises(ImportError) as excinfo:
            LLM(model="anthropic/claude-3-5-sonnet-20241022")

        assert "Error importing native provider" in str(excinfo.value)
        assert "Native Anthropic SDK failed" in str(excinfo.value)


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


def test_anthropic_stop_sequences_sync():
    """Test that stop and stop_sequences attributes stay synchronized."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Test setting stop as a list
    llm.stop = ["\nObservation:", "\nThought:"]
    assert llm.stop_sequences == ["\nObservation:", "\nThought:"]
    assert llm.stop == ["\nObservation:", "\nThought:"]

    # Test setting stop as a string
    llm.stop = "\nFinal Answer:"
    assert llm.stop_sequences == ["\nFinal Answer:"]
    assert llm.stop == ["\nFinal Answer:"]

    # Test setting stop as None
    llm.stop = None
    assert llm.stop_sequences == []
    assert llm.stop == []


@pytest.mark.vcr()
def test_anthropic_stop_sequences_sent_to_api():
    """Test that stop_sequences are properly sent to the Anthropic API."""
    llm = LLM(model="anthropic/claude-3-5-haiku-20241022")

    llm.stop = ["\nObservation:", "\nThought:"]

    result = llm.call("Say hello in one word")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_anthropic_thinking():
    """Test that thinking is properly handled and thinking params are passed to messages.create"""
    from unittest.mock import patch
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000
    )

    assert isinstance(llm, AnthropicCompletion)

    original_create = llm.client.messages.create
    captured_params = {}

    def capture_and_call(**kwargs):
        captured_params.update(kwargs)
        return original_create(**kwargs)

    with patch.object(llm.client.messages, 'create', side_effect=capture_and_call):
        result = llm.call("What is the weather in Tokyo?")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        assert "thinking" in captured_params
        assert captured_params["thinking"] == {"type": "enabled", "budget_tokens": 5000}

        assert captured_params["model"] == "claude-sonnet-4-5"
        assert captured_params["max_tokens"] == 10000
        assert "messages" in captured_params
        assert len(captured_params["messages"]) > 0


@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_anthropic_thinking_blocks_preserved_across_turns():
    """Test that thinking blocks are stored and included in subsequent API calls across turns"""
    from unittest.mock import patch
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000
    )

    assert isinstance(llm, AnthropicCompletion)

    # Capture all messages.create calls to verify thinking blocks are included
    original_create = llm.client.messages.create
    captured_calls = []

    def capture_and_call(**kwargs):
        captured_calls.append(kwargs)
        return original_create(**kwargs)

    with patch.object(llm.client.messages, 'create', side_effect=capture_and_call):
        # First call - establishes context and generates thinking blocks
        messages = [{"role": "user", "content": "What is 2+2?"}]
        first_result = llm.call(messages)

        # Verify first call completed
        assert first_result is not None
        assert isinstance(first_result, str)
        assert len(first_result) > 0

        # Verify thinking blocks were stored after first response
        assert len(llm.previous_thinking_blocks) > 0, "No thinking blocks stored after first call"
        first_thinking = llm.previous_thinking_blocks[0]
        assert first_thinking["type"] == "thinking"
        assert "thinking" in first_thinking
        assert "signature" in first_thinking

        # Store the thinking block content for comparison
        stored_thinking_content = first_thinking["thinking"]
        stored_signature = first_thinking["signature"]

        # Second call - should include thinking blocks from first call
        messages.append({"role": "assistant", "content": first_result})
        messages.append({"role": "user", "content": "Now what is 3+3?"})
        second_result = llm.call(messages)

        # Verify second call completed
        assert second_result is not None
        assert isinstance(second_result, str)

        # Verify at least 2 API calls were made
        assert len(captured_calls) >= 2, f"Expected at least 2 API calls, got {len(captured_calls)}"

        # Verify second call includes thinking blocks in assistant message
        second_call_messages = captured_calls[1]["messages"]

        # Should have: user message + assistant message (with thinking blocks) + follow-up user message
        assert len(second_call_messages) >= 2

        # Find the assistant message in the second call
        assistant_message = None
        for msg in second_call_messages:
            if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                assistant_message = msg
                break

        assert assistant_message is not None, "Assistant message with list content not found in second call"
        assert isinstance(assistant_message["content"], list)

        # Verify thinking block is included in assistant message content
        thinking_found = False
        for block in assistant_message["content"]:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_found = True
                assert "thinking" in block
                assert "signature" in block
                # Verify it matches what was stored from the first call
                assert block["thinking"] == stored_thinking_content
                assert block["signature"] == stored_signature
                break

        assert thinking_found, "Thinking block not found in assistant message content in second call"

@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_anthropic_function_calling():
    """Test that function calling is properly handled"""
    llm = LLM(model="anthropic/claude-sonnet-4-5")

    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny and 72°F"

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    result = llm.call(
        "What is the weather in Tokyo? Use the get_weather tool.",
        tools=tools,
        available_functions={"get_weather": get_weather}
    )

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    # Verify the response includes information about Tokyo's weather
    assert "tokyo" in result.lower() or "72" in result


# =============================================================================
# Agent Kickoff Structured Output Tests
# =============================================================================


@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_anthropic_tool_execution_with_available_functions():
    """
    Test that Anthropic provider correctly executes tools when available_functions is provided.

    This specifically tests the fix for double llm_call_completed emission - when
    available_functions is provided, _handle_tool_execution is called which already
    emits llm_call_completed, so the caller should not emit it again.

    The test verifies:
    1. The tool is called with correct arguments
    2. The tool result is returned directly (not wrapped in conversation)
    3. The result is valid JSON matching the tool output format
    """
    import json

    llm = LLM(model="anthropic/claude-3-5-haiku-20241022")

    # Simple tool that returns a formatted string
    def create_reasoning_plan(plan: str, steps: list, ready: bool) -> str:
        """Create a reasoning plan with steps."""
        return json.dumps({"plan": plan, "steps": steps, "ready": ready})

    tools = [
        {
            "name": "create_reasoning_plan",
            "description": "Create a structured reasoning plan for completing a task",
            "input_schema": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "High-level plan description"
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of steps to execute"
                    },
                    "ready": {
                        "type": "boolean",
                        "description": "Whether the plan is ready to execute"
                    }
                },
                "required": ["plan", "steps", "ready"]
            }
        }
    ]

    result = llm.call(
        messages=[{"role": "user", "content": "Create a simple plan to say hello. Use the create_reasoning_plan tool."}],
        tools=tools,
        available_functions={"create_reasoning_plan": create_reasoning_plan}
    )

    # Verify result is valid JSON from the tool
    assert result is not None
    assert isinstance(result, str)

    # Parse the result to verify it's valid JSON
    parsed_result = json.loads(result)
    assert "plan" in parsed_result
    assert "steps" in parsed_result
    assert "ready" in parsed_result


@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_anthropic_tool_execution_returns_tool_result_directly():
    """
    Test that when available_functions is provided, the tool result is returned directly
    without additional LLM conversation (matching OpenAI behavior for reasoning_handler).
    """
    llm = LLM(model="anthropic/claude-3-5-haiku-20241022")

    call_count = 0

    def simple_calculator(operation: str, a: int, b: int) -> str:
        """Perform a simple calculation."""
        nonlocal call_count
        call_count += 1
        if operation == "add":
            return str(a + b)
        elif operation == "multiply":
            return str(a * b)
        return "Unknown operation"

    tools = [
        {
            "name": "simple_calculator",
            "description": "Perform simple math operations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "multiply"],
                        "description": "The operation to perform"
                    },
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    ]

    result = llm.call(
        messages=[{"role": "user", "content": "Calculate 5 + 3 using the simple_calculator tool with operation 'add'."}],
        tools=tools,
        available_functions={"simple_calculator": simple_calculator}
    )

    # Tool should have been called exactly once
    assert call_count == 1, f"Expected tool to be called once, got {call_count}"

    # Result should be the direct tool output
    assert result == "8", f"Expected '8' but got '{result}'"


@pytest.mark.vcr()
def test_anthropic_agent_kickoff_structured_output_without_tools():
    """
    Test that agent kickoff returns structured output without tools.
    This tests native structured output handling for Anthropic models.
    """
    from pydantic import BaseModel, Field

    class AnalysisResult(BaseModel):
        """Structured output for analysis results."""

        topic: str = Field(description="The topic analyzed")
        key_points: list[str] = Field(description="Key insights from the analysis")
        summary: str = Field(description="Brief summary of findings")

    agent = Agent(
        role="Analyst",
        goal="Provide structured analysis on topics",
        backstory="You are an expert analyst who provides clear, structured insights.",
        llm=LLM(model="anthropic/claude-3-5-haiku-20241022"),
        tools=[],
        verbose=True,
    )

    result = agent.kickoff(
        messages="Analyze the benefits of remote work briefly. Keep it concise.",
        response_format=AnalysisResult,
    )

    assert result.pydantic is not None, "Expected pydantic output but got None"
    assert isinstance(result.pydantic, AnalysisResult), f"Expected AnalysisResult but got {type(result.pydantic)}"
    assert result.pydantic.topic, "Topic should not be empty"
    assert len(result.pydantic.key_points) > 0, "Should have at least one key point"
    assert result.pydantic.summary, "Summary should not be empty"


@pytest.mark.vcr()
def test_anthropic_agent_kickoff_structured_output_with_tools():
    """
    Test that agent kickoff returns structured output after using tools.
    This tests post-tool-call structured output handling for Anthropic models.
    """
    from pydantic import BaseModel, Field
    from crewai.tools import tool

    class CalculationResult(BaseModel):
        """Structured output for calculation results."""

        operation: str = Field(description="The mathematical operation performed")
        result: int = Field(description="The result of the calculation")
        explanation: str = Field(description="Brief explanation of the calculation")

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together and return the sum."""
        return a + b

    agent = Agent(
        role="Calculator",
        goal="Perform calculations using available tools",
        backstory="You are a calculator assistant that uses tools to compute results.",
        llm=LLM(model="anthropic/claude-3-5-haiku-20241022"),
        tools=[add_numbers],
        verbose=True,
    )

    result = agent.kickoff(
        messages="Calculate 15 + 27 using your add_numbers tool. Report the result.",
        response_format=CalculationResult,
    )

    assert result.pydantic is not None, "Expected pydantic output but got None"
    assert isinstance(result.pydantic, CalculationResult), f"Expected CalculationResult but got {type(result.pydantic)}"
    assert result.pydantic.result == 42, f"Expected result 42 but got {result.pydantic.result}"
    assert result.pydantic.operation, "Operation should not be empty"
    assert result.pydantic.explanation, "Explanation should not be empty"



@pytest.mark.vcr()
def test_anthropic_cached_prompt_tokens():
    """
    Test that Anthropic correctly extracts and tracks cached_prompt_tokens
    from cache_read_input_tokens. Uses cache_control to enable prompt caching
    and sends the same large prompt twice so the second call hits the cache.
    """
    # Anthropic requires cache_control blocks and >=1024 tokens for caching
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant. {padding}"

    llm = LLM(model="anthropic/claude-sonnet-4-5-20250929")

    def _ephemeral_user(text: str):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    # First call: creates the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say hello in one word.")},
    ])

    # Second call: same system prompt should hit the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say goodbye in one word.")},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0


@pytest.mark.vcr()
def test_anthropic_streaming_cached_prompt_tokens():
    """
    Test that Anthropic streaming correctly extracts and tracks cached_prompt_tokens.
    """
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant. {padding}"

    llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", stream=True)

    def _ephemeral_user(text: str):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    # First call: creates the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say hello in one word.")},
    ])

    # Second call: same system prompt should hit the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say goodbye in one word.")},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0


@pytest.mark.vcr()
def test_anthropic_cached_prompt_tokens_with_tools():
    """
    Test that Anthropic correctly tracks cached_prompt_tokens when tools are used.
    The large system prompt should be cached across tool-calling requests.
    """
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant that uses tools. {padding}"

    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny and 72°F"

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"],
            },
        }
    ]

    llm = LLM(model="anthropic/claude-sonnet-4-5-20250929")

    def _ephemeral_user(text: str):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    # First call with tool: creates the cache
    llm.call(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": _ephemeral_user("What is the weather in Tokyo?")},
        ],
        tools=tools,
        available_functions={"get_weather": get_weather},
    )

    # Second call with same system prompt + tools: should hit the cache
    llm.call(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": _ephemeral_user("What is the weather in Paris?")},
        ],
        tools=tools,
        available_functions={"get_weather": get_weather},
    )

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0


def test_anthropic_empty_message_content_filtered():
    """
    Test that messages with empty content are filtered out to prevent API errors.
    
    Anthropic API requires all messages to have non-empty content except for
    the optional final assistant message. This test verifies that empty user
    messages are properly filtered out.
    
    Regression test for issue #4427.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    # Test with empty user message content
    messages = [{"role": "user", "content": ""}]
    formatted, system = llm._format_messages_for_anthropic(messages)

    # Empty user message should be filtered out, but a default "Hello" should be added
    # because the message list would be empty after filtering
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"


def test_anthropic_empty_string_message_filtered():
    """
    Test that whitespace-only messages are also filtered out.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    # Test with whitespace-only content
    messages = [
        {"role": "user", "content": "   "},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    formatted, system = llm._format_messages_for_anthropic(messages)

    # Whitespace-only message should be filtered, leaving only the valid message
    assert len(formatted) == 1
    assert formatted[0]["content"] == "Hello, how are you?"


def test_anthropic_mixed_empty_and_valid_messages():
    """
    Test that valid messages are preserved when mixed with empty messages.
    
    Note: After filtering, consecutive same-role messages may have a placeholder
    assistant message inserted to maintain role alternation per Anthropic API requirements.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": ""},  # Empty assistant - should be filtered (not last)
        {"role": "user", "content": ""},  # Empty user - should be filtered
        {"role": "user", "content": "Second message"},
    ]
    formatted, system = llm._format_messages_for_anthropic(messages)

    # After filtering empty messages, we have two consecutive user messages.
    # The role alternation fix inserts a placeholder assistant message between them.
    assert len(formatted) == 3
    assert formatted[0]["content"] == "First message"
    assert formatted[1]["role"] == "assistant"  # Placeholder for alternation
    assert formatted[2]["content"] == "Second message"


def test_anthropic_final_assistant_empty_allowed():
    """
    Test that empty content is allowed for the final assistant message.
    
    Anthropic API allows the optional final assistant message to have empty content.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": ""},  # Empty final assistant - should be kept
    ]
    formatted, system = llm._format_messages_for_anthropic(messages)

    # Final assistant message with empty content should be preserved
    assert len(formatted) == 2
    assert formatted[0]["role"] == "user"
    assert formatted[1]["role"] == "assistant"
    assert formatted[1]["content"] == ""


def test_anthropic_final_assistant_trailing_whitespace_stripped():
    """
    Test that trailing whitespace is stripped from the final assistant message.
    
    Anthropic API rejects requests where the final assistant message ends with
    trailing whitespace with error: "final assistant content cannot end with
    trailing whitespace". This test verifies the fix for issue #4413.
    
    See: https://github.com/crewAIInc/crewAI/issues/4413
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    # Test case from the issue: assistant message with trailing space
    messages = [
        {"role": "user", "content": "Hello. Say world"},
        {"role": "assistant", "content": "Say: "},  # trailing space triggers the error
    ]
    formatted, system = llm._format_messages_for_anthropic(messages)

    # Trailing whitespace should be stripped from the final assistant message
    assert len(formatted) == 2
    assert formatted[0]["role"] == "user"
    assert formatted[1]["role"] == "assistant"
    assert formatted[1]["content"] == "Say:"  # No trailing whitespace
    assert not formatted[1]["content"].endswith(" ")


def test_anthropic_final_assistant_multiple_trailing_whitespace_stripped():
    """
    Test that multiple trailing whitespace characters are stripped.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    # Test with multiple trailing whitespace characters (spaces, tabs, newlines)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Response  \t\n"},  # Multiple trailing whitespace
    ]
    formatted, system = llm._format_messages_for_anthropic(messages)

    assert formatted[1]["content"] == "Response"
    assert not formatted[1]["content"][-1].isspace()


def test_anthropic_non_final_assistant_whitespace_preserved():
    """
    Test that whitespace in non-final assistant messages is preserved.
    
    Only the final assistant message needs trailing whitespace stripped.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "First response "},  # Non-final: whitespace kept
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "Second response "},  # Final: whitespace stripped
    ]
    formatted, system = llm._format_messages_for_anthropic(messages)

    assert len(formatted) == 4
    # Non-final assistant message - whitespace preserved
    assert formatted[1]["content"] == "First response "
    # Final assistant message - whitespace stripped
    assert formatted[3]["content"] == "Second response"


def test_anthropic_user_final_message_whitespace_preserved():
    """
    Test that trailing whitespace in a final user message is preserved.
    
    The stripping only applies to final assistant messages, not user messages.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022", api_key="test-key")

    messages = [
        {"role": "user", "content": "Hello with trailing space "},
    ]
    formatted, system = llm._format_messages_for_anthropic(messages)

    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    # User message whitespace should be preserved
    assert formatted[0]["content"] == "Hello with trailing space "
