import os
import sys
import types
from unittest.mock import AsyncMock, patch, MagicMock
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

    if module_name in sys.modules:
        del sys.modules[module_name]

    LLM(model="anthropic/claude-3-5-sonnet-20241022")

    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    assert hasattr(completion_mod, 'AnthropicCompletion')


def test_native_anthropic_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native Anthropic completion fails to initialize.
    This ensures we don't silently fall back when there's a configuration issue.
    """
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
    assert llm._client.max_retries == 5
    assert llm._client.timeout == 60


def test_anthropic_completion_call():
    """
    Test that AnthropicCompletion call method works
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    with patch.object(llm, 'call', return_value="Hello! I'm Claude, ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm Claude, ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


def test_anthropic_completion_called_during_crew_execution():
    """
    Test that AnthropicCompletion.call is actually invoked when running a crew
    """
    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    with patch.object(anthropic_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:

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

        assert mock_call.called
        assert "14 million" in str(result)


def test_anthropic_completion_call_arguments():
    """
    Test that AnthropicCompletion.call is invoked with correct arguments
    """
    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    with patch.object(anthropic_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed successfully."

        agent = Agent(
            role="Test Agent",
            goal="Complete a simple task",
            backstory="You are a test agent.",
            llm=anthropic_llm
        )

        task = Task(
            description="Say hello world",
            expected_output="Hello world",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()

        assert mock_call.called

        call_args = mock_call.call_args
        assert call_args is not None

        messages = call_args[0][0]
        assert isinstance(messages, (str, list))

        if isinstance(messages, str):
            assert "hello world" in messages.lower()
        elif isinstance(messages, list):
            message_content = str(messages).lower()
            assert "hello world" in message_content


def test_multiple_anthropic_calls_in_crew():
    """
    Test that AnthropicCompletion.call is invoked multiple times for multiple tasks
    """
    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    with patch.object(anthropic_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent = Agent(
            role="Multi-task Agent",
            goal="Complete multiple tasks",
            backstory="You can handle multiple tasks.",
            llm=anthropic_llm
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

        assert mock_call.call_count >= 2  # At least one call per task

        for call in mock_call.call_args_list:
            assert len(call[0]) > 0
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

    anthropic_llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    with patch.object(anthropic_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed with tools."

        agent = Agent(
            role="Tool User",
            goal="Use tools to complete tasks",
            backstory="You can use tools.",
            llm=anthropic_llm,
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
        "timeout": 120,
        "max_retries": 10,
        "default_headers": {"X-Override": "true"}
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

    assert context_size > 100000
    assert context_size <= 200000  # But not exceed the actual limit


def test_anthropic_message_formatting():
    """
    Test that messages are properly formatted for Anthropic API
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

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
    assert len(formatted_messages) >= 3


def test_anthropic_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    llm_no_stream = LLM(model="anthropic/claude-3-5-sonnet-20241022", stream=False)
    assert llm_no_stream.stream == False

    llm_stream = LLM(model="anthropic/claude-3-5-sonnet-20241022", stream=True)
    assert llm_stream.stream == True


def test_anthropic_tool_conversion():
    """
    Test that tools are properly converted to Anthropic format
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

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

        assert llm._client is not None
        assert hasattr(llm._client, 'messages')


def test_anthropic_token_usage_tracking():
    """
    Test that token usage is properly tracked for Anthropic responses
    """
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Mock the Anthropic response with usage information
    with patch.object(llm._client.messages, 'create') as mock_create:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test response")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=25)
        mock_create.return_value = mock_response

        result = llm.call("Hello")

        assert result == "test response"

        usage = llm._extract_anthropic_token_usage(mock_response)
        assert usage["input_tokens"] == 50
        assert usage["output_tokens"] == 25
        assert usage["total_tokens"] == 75


def test_anthropic_stop_sequences_sync():
    """Test that stop and stop_sequences attributes stay synchronized."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    llm.stop = ["\nObservation:", "\nThought:"]
    assert llm.stop_sequences == ["\nObservation:", "\nThought:"]
    assert llm.stop == ["\nObservation:", "\nThought:"]

    llm.stop = "\nFinal Answer:"
    assert llm.stop_sequences == ["\nFinal Answer:"]
    assert llm.stop == ["\nFinal Answer:"]

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

    original_create = llm._client.messages.create
    captured_params = {}

    def capture_and_call(**kwargs):
        captured_params.update(kwargs)
        return original_create(**kwargs)

    with patch.object(llm._client.messages, 'create', side_effect=capture_and_call):
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

    original_create = llm._client.messages.create
    captured_calls = []

    def capture_and_call(**kwargs):
        captured_calls.append(kwargs)
        return original_create(**kwargs)

    with patch.object(llm._client.messages, 'create', side_effect=capture_and_call):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        first_result = llm.call(messages)

        assert first_result is not None
        assert isinstance(first_result, str)
        assert len(first_result) > 0

        assert len(llm._previous_thinking_blocks) > 0, "No thinking blocks stored after first call"
        first_thinking = llm._previous_thinking_blocks[0]
        assert first_thinking["type"] == "thinking"
        assert "thinking" in first_thinking
        assert "signature" in first_thinking

        stored_thinking_content = first_thinking["thinking"]
        stored_signature = first_thinking["signature"]

        messages.append({"role": "assistant", "content": first_result})
        messages.append({"role": "user", "content": "Now what is 3+3?"})
        second_result = llm.call(messages)

        assert second_result is not None
        assert isinstance(second_result, str)

        assert len(captured_calls) >= 2, f"Expected at least 2 API calls, got {len(captured_calls)}"

        second_call_messages = captured_calls[1]["messages"]

        # Should have: user message + assistant message (with thinking blocks) + follow-up user message
        assert len(second_call_messages) >= 2

        assistant_message = None
        for msg in second_call_messages:
            if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                assistant_message = msg
                break

        assert assistant_message is not None, "Assistant message with list content not found in second call"
        assert isinstance(assistant_message["content"], list)

        thinking_found = False
        for block in assistant_message["content"]:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_found = True
                assert "thinking" in block
                assert "signature" in block
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
    assert "tokyo" in result.lower() or "72" in result


# Agent Kickoff Structured Output Tests


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

    assert result is not None
    assert isinstance(result, str)

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
        llm=LLM(model="anthropic/claude-sonnet-4-6"),
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

    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say hello in one word.")},
    ])

    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say goodbye in one word.")},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.successful_requests == 2
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

    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say hello in one word.")},
    ])

    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say goodbye in one word.")},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.successful_requests == 2
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

    llm.call(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": _ephemeral_user("What is the weather in Tokyo?")},
        ],
        tools=tools,
        available_functions={"get_weather": get_weather},
    )

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
    assert usage.cached_prompt_tokens > 0




def test_tool_search_true_injects_bm25_and_defer_loading():
    """tool_search=True should inject bm25 tool search and defer all tools."""
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform math calculations",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, crewai_tools
    )

    tools = params["tools"]
    assert len(tools) == 3

    assert tools[0]["type"] == "tool_search_tool_bm25_20251119"
    assert tools[0]["name"] == "tool_search_tool_bm25"
    assert "input_schema" not in tools[0]

    for t in tools[1:]:
        assert t.get("defer_loading") is True, f"Tool {t['name']} missing defer_loading"


def test_tool_search_regex_config():
    """tool_search with regex config should use regex variant."""
    from crewai.llms.providers.anthropic.completion import AnthropicToolSearchConfig

    config = AnthropicToolSearchConfig(type="regex")
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=config)

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "tool_a",
                "description": "First tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_b",
                "description": "Second tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, crewai_tools
    )

    tools = params["tools"]
    assert tools[0]["type"] == "tool_search_tool_regex_20251119"
    assert tools[0]["name"] == "tool_search_tool_regex"


def test_tool_search_disabled_by_default():
    """tool_search=None (default) should NOT inject anything."""
    llm = LLM(model="anthropic/claude-sonnet-4-5")

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, crewai_tools
    )

    tools = params["tools"]
    assert len(tools) == 1
    for t in tools:
        assert t.get("type", "") not in (
            "tool_search_tool_bm25_20251119",
            "tool_search_tool_regex_20251119",
        )
        assert "defer_loading" not in t


def test_tool_search_no_duplicate_when_manually_provided():
    """If user passes a tool search tool manually, don't inject a duplicate."""
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)

    # User manually includes a tool search tool
    tools_with_search = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, tools_with_search
    )

    tools = params["tools"]
    search_tools = [
        t for t in tools
        if t.get("type", "").startswith("tool_search_tool")
    ]
    assert len(search_tools) == 1
    assert search_tools[0]["type"] == "tool_search_tool_regex_20251119"


def test_tool_search_passthrough_preserves_tool_search_type():
    """_convert_tools_for_interference should pass through tool search tools unchanged."""
    llm = LLM(model="anthropic/claude-sonnet-4-5")

    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    ]

    converted = llm._convert_tools_for_interference(tools)
    assert len(converted) == 2
    # Tool search tool should be passed through exactly
    assert converted[0] == {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }
    # Regular tool should be preserved
    assert converted[1]["name"] == "get_weather"
    assert "input_schema" in converted[1]


def test_tool_search_single_tool_skips_search_and_forces_choice():
    """With only 1 tool, tool_search is skipped (nothing to search) and the
    normal forced tool_choice optimisation still applies."""
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages,
        system_message,
        crewai_tools,
        available_functions={"test_tool": lambda q: "result"},
    )

    # Single tool — tool_search skipped, tool_choice forced as normal
    assert "tool_choice" in params
    assert params["tool_choice"]["name"] == "test_tool"

    # No tool search tool should be injected
    tool_types = [t.get("type", "") for t in params["tools"]]
    for ts_type in ("tool_search_tool_bm25_20251119", "tool_search_tool_regex_20251119"):
        assert ts_type not in tool_types

    # No defer_loading on the single tool
    assert "defer_loading" not in params["tools"][0]


def test_tool_search_via_llm_class():
    """Verify tool_search param passes through LLM class correctly."""
    from crewai.llms.providers.anthropic.completion import (
        AnthropicCompletion,
        AnthropicToolSearchConfig,
    )

    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)
    assert isinstance(llm, AnthropicCompletion)
    assert llm.tool_search is not None
    assert llm.tool_search.type == "bm25"

    llm2 = LLM(
        model="anthropic/claude-sonnet-4-5",
        tool_search=AnthropicToolSearchConfig(type="regex"),
    )
    assert llm2.tool_search is not None
    assert llm2.tool_search.type == "regex"

    llm3 = LLM(model="anthropic/claude-sonnet-4-5")
    assert llm3.tool_search is None


# Many tools shared by the VCR tests below
_MANY_TOOLS = [
    {
        "name": name,
        "description": desc,
        "input_schema": {
            "type": "object",
            "properties": {"input": {"type": "string", "description": f"Input for {name}"}},
            "required": ["input"],
        },
    }
    for name, desc in [
        ("get_weather", "Get current weather conditions for a specified location"),
        ("search_files", "Search through files in the workspace by name or content"),
        ("read_database", "Read records from a database table with optional filtering"),
        ("write_database", "Write or update records in a database table"),
        ("send_email", "Send an email message to one or more recipients"),
        ("read_email", "Read emails from inbox with filtering options"),
        ("create_ticket", "Create a new support ticket in the ticketing system"),
        ("update_ticket", "Update an existing support ticket status or description"),
        ("list_users", "List all users in the system with optional filters"),
        ("get_user_profile", "Get detailed profile information for a specific user"),
        ("deploy_service", "Deploy a service to the specified environment"),
        ("rollback_service", "Rollback a service deployment to a previous version"),
        ("get_service_logs", "Get service logs filtered by time range and severity"),
        ("run_sql_query", "Run a read-only SQL query against the analytics database"),
        ("create_dashboard", "Create a new monitoring dashboard with widgets"),
    ]
]


def _dict_tool_use_response():
    mock_response = MagicMock()
    mock_response.content = [
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "search_web",
            "input": {"query": "CrewAI"},
        }
    ]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=2)
    mock_response.stop_reason = "tool_use"
    mock_response.id = "msg_123"
    return mock_response


class _SyncAnthropicStream:
    def __init__(self, events, final_message):
        self.events = events
        self.final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final_message


class _AsyncAnthropicStream:
    def __init__(self, events, final_message):
        self.events = list(events)
        self.final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def __aiter__(self):
        self._iter = iter(self.events)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def get_final_message(self):
        return self.final_message


def _dict_tool_use_stream_events():
    return [
        types.SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block={
                "type": "tool_use",
                "id": "toolu_123",
                "name": "search_web",
                "input": {},
            },
        ),
        types.SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=types.SimpleNamespace(
                type="input_json_delta",
                partial_json='{"query":"CrewAI"}',
            ),
        ),
    ]


def test_anthropic_tool_use_dict_blocks_are_returned_as_tool_calls():
    """Preview Anthropic models may return dict-shaped tool_use blocks."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5")
    mock_response = _dict_tool_use_response()

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    llm._client = mock_client

    result = llm.call("Search for CrewAI", tools=_MANY_TOOLS)

    assert result == mock_response.content


def test_anthropic_dict_tool_use_blocks_require_id():
    """Incomplete dict-shaped tool_use blocks are not valid tool calls."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5")
    mock_response = _dict_tool_use_response()
    del mock_response.content[0]["id"]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    llm._client = mock_client

    result = llm.call("Search for CrewAI", tools=_MANY_TOOLS)

    assert result == ""


def test_anthropic_object_tool_use_blocks_require_id():
    """Incomplete object-shaped tool_use blocks are not valid tool calls."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5")
    mock_response = MagicMock()
    mock_response.content = [
        types.SimpleNamespace(
            type="tool_use",
            name="search_web",
            input={"query": "CrewAI"},
        )
    ]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=2)
    mock_response.stop_reason = "tool_use"
    mock_response.id = "msg_123"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    llm._client = mock_client

    result = llm.call("Search for CrewAI", tools=_MANY_TOOLS)

    assert result == ""


@pytest.mark.asyncio
async def test_anthropic_acall_returns_dict_tool_use_blocks_as_tool_calls():
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5")
    mock_response = _dict_tool_use_response()

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    llm._async_client = mock_client

    result = await llm.acall("Search for CrewAI", tools=_MANY_TOOLS)

    assert result == mock_response.content


def test_anthropic_streaming_returns_dict_tool_use_blocks_as_tool_calls():
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5", stream=True)
    mock_response = _dict_tool_use_response()
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _SyncAnthropicStream(
        _dict_tool_use_stream_events(), mock_response
    )
    llm._client = mock_client
    llm._emit_stream_chunk_event = MagicMock()

    result = llm.call("Search for CrewAI", tools=_MANY_TOOLS)

    assert result == mock_response.content
    assert any(
        call.kwargs.get("tool_call", {}).get("id") == "toolu_123"
        for call in llm._emit_stream_chunk_event.call_args_list
    )


@pytest.mark.asyncio
async def test_anthropic_async_streaming_returns_dict_tool_use_blocks_as_tool_calls():
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5", stream=True)
    mock_response = _dict_tool_use_response()
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _AsyncAnthropicStream(
        _dict_tool_use_stream_events(), mock_response
    )
    llm._async_client = mock_client
    llm._emit_stream_chunk_event = MagicMock()

    result = await llm.acall("Search for CrewAI", tools=_MANY_TOOLS)

    assert result == mock_response.content
    assert any(
        call.kwargs.get("tool_call", {}).get("id") == "toolu_123"
        for call in llm._emit_stream_chunk_event.call_args_list
    )


def test_anthropic_dict_tool_use_blocks_execute_available_function():
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5")
    mock_response = _dict_tool_use_response()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    llm._client = mock_client

    result = llm.call(
        "Search for CrewAI",
        tools=_MANY_TOOLS,
        available_functions={"search_web": lambda query: f"found {query}"},
    )

    assert result == "found CrewAI"


def test_anthropic_dict_tool_use_blocks_work_in_follow_up_conversation():
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5")
    initial_response = _dict_tool_use_response()
    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="Final answer")]
    final_response.usage = MagicMock(input_tokens=4, output_tokens=3)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_final"
    mock_client = MagicMock()
    mock_client.messages.create.return_value = final_response
    llm._client = mock_client

    result = llm._handle_tool_use_conversation(
        initial_response,
        initial_response.content,
        params={"messages": []},
        available_functions={"search_web": lambda query: f"found {query}"},
    )

    assert result == "Final answer"


@pytest.mark.vcr()
def test_tool_search_discovers_and_calls_tool():
    """Tool search should discover the right tool and return a tool_use block."""
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)

    result = llm.call(
        "What is the weather in Tokyo?",
        tools=_MANY_TOOLS,
    )

    # Should return tool_use blocks (list) since no available_functions provided
    assert isinstance(result, list)
    assert len(result) >= 1
    tool_names = [getattr(block, "name", None) for block in result]
    assert "get_weather" in tool_names


@pytest.mark.vcr()
def test_tool_search_saves_input_tokens():
    """Tool search with deferred loading should use fewer input tokens than loading all tools."""
    # Call WITHOUT tool search — all 15 tools loaded upfront
    llm_no_search = LLM(model="anthropic/claude-sonnet-4-5")
    llm_no_search.call("What is the weather in Tokyo?", tools=_MANY_TOOLS)
    usage_no_search = llm_no_search.get_token_usage_summary()

    # Call WITH tool search — tools deferred
    llm_search = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)
    llm_search.call("What is the weather in Tokyo?", tools=_MANY_TOOLS)
    usage_search = llm_search.get_token_usage_summary()

    # Tool search should use fewer input tokens
    assert usage_search.prompt_tokens < usage_no_search.prompt_tokens, (
        f"Expected tool_search ({usage_search.prompt_tokens}) to use fewer input tokens "
        f"than no search ({usage_no_search.prompt_tokens})"
    )


def test_anthropic_cache_creation_tokens_extraction():
    """Test that cache_creation_input_tokens are extracted from Anthropic responses."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="test response")]
    mock_response.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=20,
    )
    mock_response.stop_reason = None
    mock_response.model = None

    usage = llm._extract_anthropic_token_usage(mock_response)
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 150
    assert usage["cached_prompt_tokens"] == 30
    assert usage["cache_creation_tokens"] == 20


def test_anthropic_missing_cache_fields_default_to_zero():
    """Test that missing cache fields default to zero."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="test response")]
    mock_response.usage = MagicMock(
        input_tokens=40,
        output_tokens=20,
        spec=["input_tokens", "output_tokens"],
    )
    mock_response.usage.cache_read_input_tokens = None
    mock_response.usage.cache_creation_input_tokens = None

    usage = llm._extract_anthropic_token_usage(mock_response)
    assert usage["cached_prompt_tokens"] == 0
    assert usage["cache_creation_tokens"] == 0
