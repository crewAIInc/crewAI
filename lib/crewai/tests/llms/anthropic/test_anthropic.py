import os
import sys
import types
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from pydantic import BaseModel

from crewai.llm import LLM
from crewai.llms.context_window import CONTEXT_WINDOW_USAGE_RATIO
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


def test_anthropic_completion_defaults_to_sonnet_4_6():
    from crewai.llms.providers.anthropic.completion import (
        DEFAULT_MODEL,
        AnthropicCompletion,
    )

    llm = AnthropicCompletion()
    assert llm.model == DEFAULT_MODEL
    assert llm.model == "claude-sonnet-4-6"
    assert llm.max_tokens == 128000


def test_default_anthropic_completion_uses_sonnet_4_6_context_window():
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion()
    assert llm.get_context_window_size() == int(1_000_000 * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    ],
)
def test_current_1m_anthropic_models_use_1m_context_window(model: str) -> None:
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model=model)
    assert llm.get_context_window_size() == int(1_000_000 * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
)
def test_claude_4_5_family_uses_200k_context_window(model: str) -> None:
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model=model)
    assert llm.get_context_window_size() == int(200000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_anthropic_defaults_max_tokens_to_model_limit():
    """Native Anthropic used to default to 4096, which truncates large tool calls."""
    llm = LLM(model="anthropic/claude-unknown-future")

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    assert isinstance(llm, AnthropicCompletion)
    assert llm.max_tokens == 32000
    params = llm._prepare_completion_params(messages=[{"role": "user", "content": "hi"}])
    assert params["max_tokens"] == 32000
    assert "max_tokens" not in llm.to_config_dict()

    sonnet45 = LLM(model="anthropic/claude-sonnet-4-5")
    assert isinstance(sonnet45, AnthropicCompletion)
    assert sonnet45.max_tokens == 64000
    assert "max_tokens" not in sonnet45.to_config_dict()


def test_anthropic_honors_explicit_max_tokens():
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022", max_tokens=8000)

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    assert isinstance(llm, AnthropicCompletion)
    assert llm.max_tokens == 8000
    params = llm._prepare_completion_params(messages=[{"role": "user", "content": "hi"}])
    assert params["max_tokens"] == 8000
    assert llm.to_config_dict()["max_tokens"] == 8000


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
    llm = LLM(model="anthropic/claude-haiku-4-5")
    context_size = llm.get_context_window_size()

    assert context_size == int(200000 * CONTEXT_WINDOW_USAGE_RATIO)


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
        mock_response.usage = MagicMock(
            input_tokens=50,
            output_tokens=25,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_create.return_value = mock_response

        result = llm.call("Hello")

        assert result == "test response"

        usage = llm._extract_anthropic_token_usage(mock_response)
        assert usage["input_tokens"] == 50
        assert usage["output_tokens"] == 25
        assert usage["total_tokens"] == 75


def test_anthropic_input_token_count_forwards_completion_payload():
    """Counting formats system messages and tools like a completion request."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")
    client = MagicMock()
    client.messages.count_tokens.return_value = MagicMock(input_tokens=42)
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is the weather?"},
    ]
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    with patch.object(llm, "_get_sync_client", return_value=client):
        assert llm.get_input_tokens_count(messages, tools=tools) == 42

    client.messages.count_tokens.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        system="Be concise.",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather for a location.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )


def test_anthropic_input_token_count_includes_files_api_beta():
    """File-backed messages use the same beta header as generation."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")
    client = MagicMock()
    client.beta.messages.count_tokens.return_value = MagicMock(input_tokens=42)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "file_123"},
                }
            ],
        }
    ]

    with patch.object(llm, "_get_sync_client", return_value=client):
        assert llm.get_input_tokens_count(messages) == 42

    client.beta.messages.count_tokens.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        betas=["files-api-2025-04-14"],
    )


def test_anthropic_input_token_count_uses_character_fallback():
    """Count endpoint errors fall back to the character estimate."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")
    client = MagicMock()
    client.messages.count_tokens.side_effect = RuntimeError("count failed")

    with patch.object(llm, "_get_sync_client", return_value=client):
        assert llm.get_input_tokens_count("12345") == 2


def test_anthropic_input_token_count_uses_files_beta_for_structured_output_fallback():
    """Structured output fallback retains the Files API beta header."""
    class Answer(BaseModel):
        value: str

    llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")
    client = MagicMock()
    client.beta.messages.count_tokens.return_value = MagicMock(input_tokens=42)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "file_123"},
                }
            ],
        }
    ]

    with patch.object(llm, "_get_sync_client", return_value=client):
        assert llm.get_input_tokens_count(messages, response_model=Answer) == 42

    client.beta.messages.count_tokens.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        tools=[
            {
                "name": "structured_output",
                "description": "Output the structured response",
                "input_schema": {
                    "additionalProperties": False,
                    "properties": {"value": {"title": "Value", "type": "string"}},
                    "required": ["value"],
                    "title": "Answer",
                    "type": "object",
                },
            }
        ],
        tool_choice={"type": "tool", "name": "structured_output"},
        betas=["files-api-2025-04-14"],
    )


@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_anthropic_input_token_count_live():
    """Anthropic returns a positive input-token count for a real request."""
    llm = LLM(model="anthropic/claude-haiku-4-5")

    assert llm.get_input_tokens_count("Count these input tokens.") > 0


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


def test_anthropic_thinking_blocks_preserved_on_tool_use_turn():
    """A tool_use assistant turn must keep the thinking blocks Anthropic requires"""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    tool_use_response = MagicMock()
    tool_use_response.content = [
        {
            "type": "thinking",
            "thinking": "Check the weather in Paris.",
            "signature": "sig-abc",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    tool_use_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="It is 18C in Paris.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_use_response, final_response]
    llm._client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]

    tool_calls = llm.call(messages, tools=tools)
    assert tool_calls == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ]

    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})

    llm.call(messages, tools=tools)

    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    assistant_message = next(
        msg for msg in second_call_messages if msg["role"] == "assistant"
    )
    assert isinstance(assistant_message["content"], list)
    assert assistant_message["content"][0] == {
        "type": "thinking",
        "thinking": "Check the weather in Paris.",
        "signature": "sig-abc",
    }
    assert assistant_message["content"][1]["type"] == "tool_use"
    assert assistant_message["content"][1]["id"] == "toolu_1"


def test_anthropic_thinking_blocks_only_on_matching_tool_use_turn():
    """Across two consecutive tool-use turns, thinking blocks stay on their own turn"""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    first_tool_response = MagicMock()
    first_tool_response.content = [
        {
            "type": "thinking",
            "thinking": "Look up the weather in Paris.",
            "signature": "sig-one",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    first_tool_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    first_tool_response.stop_reason = "tool_use"
    first_tool_response.id = "msg_1"

    second_tool_response = MagicMock()
    second_tool_response.content = [
        {
            "type": "thinking",
            "thinking": "Now look up the weather in Rome.",
            "signature": "sig-two",
        },
        {
            "type": "tool_use",
            "id": "toolu_2",
            "name": "get_weather",
            "input": {"city": "Rome"},
        },
    ]
    second_tool_response.usage = MagicMock(input_tokens=11, output_tokens=6)
    second_tool_response.stop_reason = "tool_use"
    second_tool_response.id = "msg_2"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="18C in Paris, 24C in Rome.")]
    final_response.usage = MagicMock(input_tokens=13, output_tokens=7)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_3"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        first_tool_response,
        second_tool_response,
        final_response,
    ]
    llm._client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris and Rome?"}]

    assert llm.call(messages, tools=tools)[0]["id"] == "toolu_1"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})

    assert llm.call(messages, tools=tools)[0]["id"] == "toolu_2"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Rome"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_2", "content": "24C"})

    llm.call(messages, tools=tools)

    # Second request: the first tool-use turn is the matching turn, so it
    # carries the thinking blocks of the response that produced toolu_1.
    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    first_turn = next(
        msg
        for msg in second_call_messages
        if msg["role"] == "assistant" and msg.get("content")
        and isinstance(msg["content"], list)
        and any(block.get("id") == "toolu_1" for block in msg["content"])
    )
    assert first_turn["content"][0] == {
        "type": "thinking",
        "thinking": "Look up the weather in Paris.",
        "signature": "sig-one",
    }

    # Third request: the latest thinking blocks belong to the second
    # tool-use turn only. The earlier turn must not receive them.
    third_call_messages = mock_client.messages.create.call_args_list[2].kwargs["messages"]
    tool_use_turns = [
        msg
        for msg in third_call_messages
        if msg["role"] == "assistant"
        and isinstance(msg.get("content"), list)
        and any(block.get("type") == "tool_use" for block in msg["content"])
    ]
    assert len(tool_use_turns) == 2

    earlier_turn, matching_turn = tool_use_turns
    assert earlier_turn["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ]
    assert matching_turn["content"][0] == {
        "type": "thinking",
        "thinking": "Now look up the weather in Rome.",
        "signature": "sig-two",
    }
    assert matching_turn["content"][1]["type"] == "tool_use"
    assert matching_turn["content"][1]["id"] == "toolu_2"


def test_anthropic_redacted_thinking_data_preserved_on_tool_use_turn():
    """A redacted thinking block must keep its opaque data on replay

    Regression: `_extract_thinking_block` copied thinking/signature for
    redacted blocks, but a RedactedThinkingBlock only carries an encrypted
    `data` field that Anthropic requires back unchanged. The cached block
    was replayed without it, making the follow-up request invalid.
    """
    from anthropic.types import RedactedThinkingBlock
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    # The SDK object shape keeps its data too.
    extracted = llm._extract_thinking_block(
        RedactedThinkingBlock(data="opaque-encrypted-data-sdk", type="redacted_thinking")
    )
    assert extracted == {
        "type": "redacted_thinking",
        "data": "opaque-encrypted-data-sdk",
    }

    tool_use_response = MagicMock()
    tool_use_response.content = [
        {"type": "redacted_thinking", "data": "opaque-encrypted-data"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    tool_use_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="It is 18C in Paris.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_use_response, final_response]
    llm._client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]

    tool_calls = llm.call(messages, tools=tools)
    assert tool_calls[0]["id"] == "toolu_1"
    assert llm._previous_thinking_blocks == [
        {"type": "redacted_thinking", "data": "opaque-encrypted-data"}
    ]

    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})

    llm.call(messages, tools=tools)

    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    assistant_message = next(
        msg for msg in second_call_messages if msg["role"] == "assistant"
    )
    redacted_block = assistant_message["content"][0]
    # The replayed block must be exactly what Anthropic handed back.
    assert redacted_block == {
        "type": "redacted_thinking",
        "data": "opaque-encrypted-data",
    }
    assert RedactedThinkingBlock.model_validate(redacted_block) is not None


def test_anthropic_thinking_blocks_not_replayed_across_conversations():
    """Cached thinking must not leak into another conversation's turns

    Regression: `_previous_thinking_blocks` was replayed on whatever
    history the LLM instance formatted next, so when one instance served
    two conversations, conversation B's assistant turns received
    conversation A's signed thinking blocks.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    tool_use_response = MagicMock()
    tool_use_response.content = [
        {
            "type": "thinking",
            "thinking": "Conversation A: check the weather in Paris.",
            "signature": "sig-A",
        },
        {
            "type": "tool_use",
            "id": "toolu_A",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    tool_use_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="Conversation B answer.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        tool_use_response,
        tool_use_response,
        final_response,
    ]
    llm._client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    # Conversation A: the tool-use turn caches A's thinking blocks.
    llm.call(
        [{"role": "user", "content": "A: what is the weather in Paris?"}], tools=tools
    )

    # Conversation B shares this LLM instance with its own history. Its
    # text assistant turn must not receive A's cached thinking blocks.
    llm.call(
        [
            {"role": "user", "content": "B: first question."},
            {"role": "assistant", "content": "B: earlier answer."},
            {"role": "user", "content": "B: second question."},
        ]
    )

    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    b_assistant_turns = [
        msg for msg in second_call_messages if msg["role"] == "assistant"
    ]
    assert len(b_assistant_turns) == 1
    assert b_assistant_turns[0]["content"] == "B: earlier answer."

    # B's tool-use turn carries different tool ids, so A's cached blocks
    # must not be prepended to it either.
    llm.call(
        [
            {"role": "user", "content": "B: first question."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_B",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_B", "content": "18C"},
        ],
        tools=tools,
    )

    third_call_messages = mock_client.messages.create.call_args_list[2].kwargs["messages"]
    b_tool_turn = next(
        msg
        for msg in third_call_messages
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list)
    )
    assert b_tool_turn["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_B",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ]


def test_anthropic_thinking_blocks_replayed_when_turn_persists_subset_of_tool_calls():
    """A turn that kept part of the tool batch must still get its thinking.

    Regression: the replay gate required the turn's tool_use ids to equal the
    cached set exactly. Executors can persist only a subset of a response's
    tool calls (``_execute_first_tool`` runs the first call, and a batch
    stops at a ``result_as_answer`` tool), so the follow-up request went out
    as a bare ``tool_use`` turn — the Anthropic 400 this PR fixes.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    tool_use_response = MagicMock()
    tool_use_response.content = [
        {
            "type": "thinking",
            "thinking": "Look up both cities.",
            "signature": "sig-1",
        },
        {
            "type": "tool_use",
            "id": "toolu_A",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
        {
            "type": "tool_use",
            "id": "toolu_B",
            "name": "get_weather",
            "input": {"city": "Rome"},
        },
    ]
    tool_use_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="Done.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_use_response, final_response]
    llm._client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    # The response asks for two tools, but the history keeps only the first
    # call, as executors running one tool or stopping at result_as_answer do.
    llm.call(
        [{"role": "user", "content": "Weather for Paris and Rome?"}], tools=tools
    )
    llm.call(
        [
            {"role": "user", "content": "Weather for Paris and Rome?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_A",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_A", "content": "18C"},
        ],
        tools=tools,
    )

    follow_up_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    a_tool_turns = [
        msg
        for msg in follow_up_messages
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list)
    ]
    assert a_tool_turns, "the subset tool turn must carry its thinking blocks"
    assert a_tool_turns[0]["content"] == [
        {
            "type": "thinking",
            "thinking": "Look up both cities.",
            "signature": "sig-1",
        },
        {
            "type": "tool_use",
            "id": "toolu_A",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]


def test_anthropic_thinking_blocks_replayed_when_text_truncated_by_stop_words():
    """A stop-word-truncated turn must still be recognized as its own.

    Regression: the text fingerprint compared the persisted turn content to
    the raw response text exactly. The provider's return value is truncated
    at stop words (and may be extended by after-call hooks), so the persisted
    turn differed from the raw text and its thinking blocks were dropped.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
        stop=["Observation:"],
    )

    assert isinstance(llm, AnthropicCompletion)

    thinking_response = MagicMock()
    thinking_response.content = [
        {
            "type": "thinking",
            "thinking": "Answer the question directly.",
            "signature": "sig-2",
        },
        types.SimpleNamespace(text="The answer. Observation: hidden tail"),
    ]
    thinking_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    thinking_response.stop_reason = "end_turn"
    thinking_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="Done.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [thinking_response, final_response]
    llm._client = mock_client

    # The executor persists the provider's truncated return value.
    llm.call([{"role": "user", "content": "What is the answer?"}])
    llm.call(
        [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "The answer."},
            {"role": "user", "content": "Continue."},
        ]
    )

    follow_up_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    text_turns = [
        msg
        for msg in follow_up_messages
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list)
    ]
    assert text_turns, "the truncated text turn must carry its thinking blocks"
    assert text_turns[0]["content"] == [
        {
            "type": "thinking",
            "thinking": "Answer the question directly.",
            "signature": "sig-2",
        },
        {"type": "text", "text": "The answer."},
    ]


def test_anthropic_thinking_blocks_not_replayed_when_other_conversations_text_is_a_prefix():
    """A text fingerprint match alone must not replay thinking blocks.

    Regression: the text gate accepted any non-empty prefix or extension
    of the cached raw text, so on a shared LLM instance a later
    conversation whose last assistant text happened to be a prefix of the
    cached response (or vice versa) received that response's signed
    thinking blocks.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    thinking_response = MagicMock()
    thinking_response.content = [
        {
            "type": "thinking",
            "thinking": "Conversation A reasoning.",
            "signature": "sig-A",
        },
        types.SimpleNamespace(text="The capital of France is Paris, its largest city."),
    ]
    thinking_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    thinking_response.stop_reason = "end_turn"
    thinking_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="B is done.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [thinking_response, final_response]
    llm._client = mock_client

    # Conversation A caches its thinking blocks with a text fingerprint.
    llm.call([{"role": "user", "content": "A: capital of France?"}])

    # Conversation B shares this instance. Its last assistant text is a
    # strict prefix of A's raw response text, but the turn belongs to a
    # different conversation and must not carry A's thinking blocks.
    llm.call(
        [
            {"role": "user", "content": "B: first question."},
            {"role": "assistant", "content": "The capital of France"},
            {"role": "user", "content": "B: second question."},
        ]
    )

    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs[
        "messages"
    ]
    b_structured_turns = [
        msg
        for msg in second_call_messages
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list)
    ]
    assert b_structured_turns == []


def test_anthropic_thinking_blocks_replayed_when_stop_words_strip_leading_whitespace():
    """Leading whitespace in the raw text must not hide a truncated turn.

    Regression: ``_apply_stop_words`` strips the truncated return value,
    and raw response text commonly opens with the whitespace that follows
    a thinking block. The exact and prefix comparisons both failed on that
    leading whitespace, so the originating turn lost its thinking blocks.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
        stop=["Observation:"],
    )

    assert isinstance(llm, AnthropicCompletion)

    thinking_response = MagicMock()
    thinking_response.content = [
        {
            "type": "thinking",
            "thinking": "Answer with the number only.",
            "signature": "sig-3",
        },
        types.SimpleNamespace(text="\n\nThe answer is 42. Observation: hidden tail"),
    ]
    thinking_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    thinking_response.stop_reason = "end_turn"
    thinking_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="Done.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [thinking_response, final_response]
    llm._client = mock_client

    # The executor persists the provider's truncated-and-stripped return
    # value, which no longer shares the raw text's leading whitespace.
    llm.call([{"role": "user", "content": "What is the answer?"}])
    llm.call(
        [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "The answer is 42."},
            {"role": "user", "content": "Continue."},
        ]
    )

    follow_up_messages = mock_client.messages.create.call_args_list[1].kwargs[
        "messages"
    ]
    text_turns = [
        msg
        for msg in follow_up_messages
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list)
    ]
    assert text_turns, "the whitespace-stripped turn must carry its thinking blocks"
    assert text_turns[0]["content"] == [
        {
            "type": "thinking",
            "thinking": "Answer with the number only.",
            "signature": "sig-3",
        },
        {"type": "text", "text": "The answer is 42."},
    ]


def test_anthropic_thinking_blocks_replayed_on_every_turn_of_a_multi_turn_conversation():
    """Multi-turn text conversations must keep replaying each turn's thinking.

    Regression: the conversation snapshot stored the replayed turn in its
    injected (structured) form, while the next request formats that same
    historical turn as plain text, so the prefix check failed and every
    other turn silently lost its thinking blocks.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    def thinking_response(signature: str, text: str, message_id: str) -> MagicMock:
        resp = MagicMock()
        resp.content = [
            {"type": "thinking", "thinking": "reasoning", "signature": signature},
            types.SimpleNamespace(text=text),
        ]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        resp.stop_reason = "end_turn"
        resp.id = message_id
        return resp

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        thinking_response("sig-1", "first answer.", "msg_1"),
        thinking_response("sig-2", "second answer.", "msg_2"),
        thinking_response("sig-3", "third answer.", "msg_3"),
    ]
    llm._client = mock_client

    llm.call([{"role": "user", "content": "first question?"}])
    llm.call(
        [
            {"role": "user", "content": "first question?"},
            {"role": "assistant", "content": "first answer."},
            {"role": "user", "content": "second question?"},
        ]
    )
    llm.call(
        [
            {"role": "user", "content": "first question?"},
            {"role": "assistant", "content": "first answer."},
            {"role": "user", "content": "second question?"},
            {"role": "assistant", "content": "second answer."},
            {"role": "user", "content": "third question?"},
        ]
    )

    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs[
        "messages"
    ]
    second_assistant = next(
        msg for msg in second_call_messages if msg["role"] == "assistant"
    )
    assert second_assistant["content"] == [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig-1"},
        {"type": "text", "text": "first answer."},
    ]

    third_call_messages = mock_client.messages.create.call_args_list[2].kwargs[
        "messages"
    ]
    assistant_turns = [msg for msg in third_call_messages if msg["role"] == "assistant"]
    # The earlier turn is no longer last, so it goes back to plain text.
    assert assistant_turns[0]["content"] == "first answer."
    # The latest turn must still carry its own thinking blocks.
    assert assistant_turns[-1]["content"] == [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig-2"},
        {"type": "text", "text": "second answer."},
    ]


def test_anthropic_stale_thinking_not_replayed_after_no_thinking_response():
    """A no-thinking follow-up must clear stored thinking for later turns

    Regression: `_previous_thinking_blocks` kept the tool turn's blocks when
    the follow-up response had no thinking, so a continued conversation
    replayed that turn's signed thinking on the latest assistant message.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    tool_use_response = MagicMock()
    tool_use_response.content = [
        {
            "type": "thinking",
            "thinking": "Check the weather in Paris.",
            "signature": "sig-abc",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    tool_use_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.id = "msg_1"

    no_thinking_response = MagicMock()
    no_thinking_response.content = [types.SimpleNamespace(text="It is 18C in Paris.")]
    no_thinking_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    no_thinking_response.stop_reason = "end_turn"
    no_thinking_response.id = "msg_2"

    follow_up_response = MagicMock()
    follow_up_response.content = [types.SimpleNamespace(text="A light jacket is enough.")]
    follow_up_response.usage = MagicMock(input_tokens=14, output_tokens=8)
    follow_up_response.stop_reason = "end_turn"
    follow_up_response.id = "msg_3"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        tool_use_response,
        no_thinking_response,
        follow_up_response,
    ]
    llm._client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]

    # Turn 1: the model thinks and requests the weather tool.
    tool_calls = llm.call(messages, tools=tools)
    assert tool_calls[0]["id"] == "toolu_1"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})

    # Turn 2: the tool result comes back and the model answers without
    # thinking, so no thinking blocks belong to the next request.
    llm.call(messages, tools=tools)
    assert llm._previous_thinking_blocks == []
    messages.append({"role": "assistant", "content": "It is 18C in Paris."})

    # Turn 3: the conversation continues; turn 1's thinking must not be
    # replayed on the latest assistant message.
    messages.append({"role": "user", "content": "What should I wear?"})
    llm.call(messages)

    third_call_messages = mock_client.messages.create.call_args_list[2].kwargs["messages"]
    for message in third_call_messages:
        if message.get("role") == "assistant" and isinstance(message.get("content"), list):
            assert not any(
                block.get("type") in ("thinking", "redacted_thinking")
                for block in message["content"]
            ), "Stale thinking blocks were replayed on a later assistant turn"


def test_anthropic_streaming_stale_thinking_not_replayed_after_no_thinking_response():
    """Streaming: a no-thinking follow-up must clear stored thinking too"""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(
        model="claude-fable-5",
        stream=True,
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    tool_use_final = MagicMock()
    tool_use_final.content = [
        {
            "type": "thinking",
            "thinking": "Check the weather in Paris.",
            "signature": "sig-abc",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    tool_use_final.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_use_final.stop_reason = "tool_use"
    tool_use_final.id = "msg_1"

    no_thinking_final = MagicMock()
    no_thinking_final.content = [types.SimpleNamespace(text="It is 18C in Paris.")]
    no_thinking_final.usage = MagicMock(input_tokens=12, output_tokens=6)
    no_thinking_final.stop_reason = "end_turn"
    no_thinking_final.id = "msg_2"

    follow_up_final = MagicMock()
    follow_up_final.content = [types.SimpleNamespace(text="A light jacket is enough.")]
    follow_up_final.usage = MagicMock(input_tokens=14, output_tokens=8)
    follow_up_final.stop_reason = "end_turn"
    follow_up_final.id = "msg_3"

    mock_client = MagicMock()
    mock_client.messages.stream.side_effect = [
        _SyncAnthropicStream([], tool_use_final),
        _SyncAnthropicStream([], no_thinking_final),
        _SyncAnthropicStream([], follow_up_final),
    ]
    llm._client = mock_client
    llm._emit_stream_chunk_event = MagicMock()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]

    # Turn 1: the model thinks and requests the weather tool.
    tool_calls = llm.call(messages, tools=tools)
    assert tool_calls[0]["id"] == "toolu_1"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})

    # Turn 2: the follow-up has no thinking, so the stored blocks are stale.
    llm.call(messages, tools=tools)
    assert llm._previous_thinking_blocks == []
    messages.append({"role": "assistant", "content": "It is 18C in Paris."})

    # Turn 3: the continued conversation must not replay turn 1's thinking.
    messages.append({"role": "user", "content": "What should I wear?"})
    llm.call(messages)

    third_call_messages = mock_client.messages.stream.call_args_list[2].kwargs["messages"]
    for message in third_call_messages:
        if message.get("role") == "assistant" and isinstance(message.get("content"), list):
            assert not any(
                block.get("type") in ("thinking", "redacted_thinking")
                for block in message["content"]
            ), "Stale thinking blocks were replayed on a later assistant turn"


def test_anthropic_tool_use_conversation_thinking_cleared_after_no_thinking_follow_up():
    """The tool-use conversation flow must also drop thinking on no-thinking replies"""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    initial_response = MagicMock()
    initial_response.content = [
        {
            "type": "thinking",
            "thinking": "Check the weather in Paris.",
            "signature": "sig-abc",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    initial_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    initial_response.stop_reason = "tool_use"
    initial_response.id = "msg_1"

    follow_up_response = MagicMock()
    follow_up_response.content = [types.SimpleNamespace(text="It is 18C in Paris.")]
    follow_up_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    follow_up_response.stop_reason = "end_turn"
    follow_up_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = follow_up_response
    llm._client = mock_client

    # A previous response already stored its thinking, as the completion
    # handler does before handing tool use over to the conversation flow.
    llm._previous_thinking_blocks = [
        {
            "type": "thinking",
            "thinking": "Check the weather in Paris.",
            "signature": "sig-abc",
        }
    ]

    result = llm._handle_tool_use_conversation(
        initial_response,
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ],
        params={
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
        },
        available_functions={"get_weather": lambda **kw: "18C"},
    )

    assert result == "It is 18C in Paris."

    # The follow-up request keeps the tool turn's thinking inline...
    sent_messages = mock_client.messages.create.call_args.kwargs["messages"]
    assistant_turn = next(
        message
        for message in sent_messages
        if message["role"] == "assistant" and isinstance(message.get("content"), list)
    )
    assert assistant_turn["content"][0] == {
        "type": "thinking",
        "thinking": "Check the weather in Paris.",
        "signature": "sig-abc",
    }

    # ...but the follow-up response carries no thinking, so the cached
    # blocks must not survive it.
    assert llm._previous_thinking_blocks == []


@pytest.mark.asyncio
async def test_anthropic_async_tool_use_turn_keeps_thinking_blocks():
    """Async: a tool-use turn must cache its thinking for the follow-up request

    Regression: `_ahandle_completion` returned tool calls without updating
    `_previous_thinking_blocks`, so the executor's follow-up `acall` sent a
    bare `tool_use` assistant turn and Anthropic rejected the request.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = LLM(
        model="anthropic/claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    assert isinstance(llm, AnthropicCompletion)

    tool_response = MagicMock()
    tool_response.content = [
        {
            "type": "thinking",
            "thinking": "Check the weather in Paris.",
            "signature": "sig-abc",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    tool_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_response.stop_reason = "tool_use"
    tool_response.id = "msg_1"

    final_response = MagicMock()
    final_response.content = [types.SimpleNamespace(text="It is 18C in Paris.")]
    final_response.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_response.stop_reason = "end_turn"
    final_response.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])
    llm._async_client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]

    tool_calls = await llm.acall(messages, tools=tools)
    assert tool_calls[0]["id"] == "toolu_1"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})

    assert await llm.acall(messages, tools=tools) == "It is 18C in Paris."

    # The follow-up request must carry the tool turn's thinking blocks on the
    # matching assistant turn, exactly like the synchronous path.
    follow_up_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    tool_turn = next(
        message
        for message in follow_up_messages
        if message["role"] == "assistant"
        and isinstance(message.get("content"), list)
        and any(block.get("id") == "toolu_1" for block in message["content"])
    )
    assert tool_turn["content"][0] == {
        "type": "thinking",
        "thinking": "Check the weather in Paris.",
        "signature": "sig-abc",
    }


@pytest.mark.asyncio
async def test_anthropic_async_streaming_tool_use_turn_keeps_thinking_blocks():
    """Async streaming: a tool-use turn must cache its thinking as well

    Regression: `_ahandle_streaming_completion` returned tool calls without
    updating `_previous_thinking_blocks`, so the executor's follow-up `acall`
    sent a bare `tool_use` assistant turn and Anthropic rejected the request.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(
        model="claude-fable-5",
        stream=True,
        thinking={"type": "enabled", "budget_tokens": 5000},
        max_tokens=10000,
    )

    tool_use_final = MagicMock()
    tool_use_final.content = [
        {
            "type": "thinking",
            "thinking": "Check the weather in Paris.",
            "signature": "sig-abc",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    tool_use_final.usage = MagicMock(input_tokens=10, output_tokens=5)
    tool_use_final.stop_reason = "tool_use"
    tool_use_final.id = "msg_1"

    final_message = MagicMock()
    final_message.content = [types.SimpleNamespace(text="It is 18C in Paris.")]
    final_message.usage = MagicMock(input_tokens=12, output_tokens=6)
    final_message.stop_reason = "end_turn"
    final_message.id = "msg_2"

    mock_client = MagicMock()
    mock_client.messages.stream.side_effect = [
        _AsyncAnthropicStream([], tool_use_final),
        _AsyncAnthropicStream([], final_message),
    ]
    llm._async_client = mock_client
    llm._emit_stream_chunk_event = MagicMock()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]

    tool_calls = await llm.acall(messages, tools=tools)
    assert tool_calls[0]["id"] == "toolu_1"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})

    await llm.acall(messages, tools=tools)

    # The follow-up request must carry the tool turn's thinking blocks on the
    # matching assistant turn, exactly like the synchronous streaming path.
    follow_up_messages = mock_client.messages.stream.call_args_list[1].kwargs["messages"]
    tool_turn = next(
        message
        for message in follow_up_messages
        if message["role"] == "assistant"
        and isinstance(message.get("content"), list)
        and any(block.get("id") == "toolu_1" for block in message["content"])
    )
    assert tool_turn["content"][0] == {
        "type": "thinking",
        "thinking": "Check the weather in Paris.",
        "signature": "sig-abc",
    }


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


def test_tool_search_single_tool_skips_search_and_uses_auto_choice():
    """With only 1 tool, tool_search is skipped (nothing to search) and
    tool_choice stays auto."""
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

    # Single tool — tool_search skipped, model chooses whether to call it
    assert params["tool_choice"] == {"type": "auto"}

    # No tool search tool should be injected
    tool_types = [t.get("type", "") for t in params["tools"]]
    for ts_type in ("tool_search_tool_bm25_20251119", "tool_search_tool_regex_20251119"):
        assert ts_type not in tool_types

    # No defer_loading on the single tool
    assert "defer_loading" not in params["tools"][0]


def test_single_tool_with_available_functions_uses_auto_choice():
    """One executable tool is offered, not forced.

    Newer Claude models reject tool_choice {"type": "tool"}. Auto lets the
    model call the tool or answer in text.
    """
    llm = LLM(model="anthropic/claude-sonnet-5-5")
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "What's the weather in Paris?"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages,
        system_message,
        [tool],
        available_functions={"get_weather": lambda city: "sunny"},
    )

    assert params["tool_choice"] == {"type": "auto"}
    assert [t["name"] for t in params["tools"]] == ["get_weather"]


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


def _assert_tool_call_completed_once(llm, mock_response):
    from crewai.events.types.llm_events import LLMCallType

    llm._emit_call_completed_event.assert_called_once()
    kwargs = llm._emit_call_completed_event.call_args.kwargs
    assert kwargs["call_type"] == LLMCallType.TOOL_CALL
    assert kwargs["response"] == mock_response.content
    assert kwargs["finish_reason"] == "tool_use"
    assert kwargs["response_id"] == "msg_123"
    assert kwargs["usage"]["output_tokens"] == 2


def test_anthropic_streaming_tool_calls_emit_call_completed_event():
    """Returned tool calls complete the call when streaming, as they do without it."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5", stream=True)
    mock_response = _dict_tool_use_response()
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _SyncAnthropicStream(
        _dict_tool_use_stream_events(), mock_response
    )
    llm._client = mock_client
    llm._emit_call_completed_event = MagicMock()

    llm.call("Search for CrewAI", tools=_MANY_TOOLS)

    _assert_tool_call_completed_once(llm, mock_response)


@pytest.mark.asyncio
async def test_anthropic_async_streaming_tool_calls_emit_call_completed_event():
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-fable-5", stream=True)
    mock_response = _dict_tool_use_response()
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _AsyncAnthropicStream(
        _dict_tool_use_stream_events(), mock_response
    )
    llm._async_client = mock_client
    llm._emit_call_completed_event = MagicMock()

    await llm.acall("Search for CrewAI", tools=_MANY_TOOLS)

    _assert_tool_call_completed_once(llm, mock_response)


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
    assert usage["input_tokens"] == 150
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 200
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


# --- Native structured outputs: model gate -----------------------------------

NATIVE_STRUCTURED_OUTPUT_MODELS_UNDER_TEST = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-haiku-4-5",
]


class _Answer(BaseModel):
    answer: str


_ANSWER_JSON = '{"answer": "42"}'

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _structured_text_response():
    """A response whose single text block holds the structured JSON payload."""
    from anthropic.types.beta import BetaTextBlock

    mock_response = MagicMock()
    mock_response.content = [
        BetaTextBlock(type="text", text=_ANSWER_JSON, citations=None)
    ]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = "end_turn"
    mock_response.id = "msg_structured"
    return mock_response


def _structured_stream_events():
    return [
        types.SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=types.SimpleNamespace(type="text_delta", text=_ANSWER_JSON),
        )
    ]


@pytest.mark.parametrize("model", NATIVE_STRUCTURED_OUTPUT_MODELS_UNDER_TEST)
def test_native_structured_output_sync(model):
    """Current Claude models ask the API for JSON directly, not via a forced tool."""
    from crewai.llms.providers.anthropic.completion import (
        ANTHROPIC_STRUCTURED_OUTPUTS_BETA,
        AnthropicCompletion,
    )

    llm = AnthropicCompletion(model=model)
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = _structured_text_response()
    llm._client = mock_client

    result = llm.call("What is the answer?", response_model=_Answer)

    assert result == _Answer(answer="42")
    kwargs = mock_client.beta.messages.create.call_args.kwargs
    assert ANTHROPIC_STRUCTURED_OUTPUTS_BETA in kwargs["betas"]
    assert kwargs["extra_body"]["output_format"]["type"] == "json_schema"
    mock_client.messages.create.assert_not_called()


@pytest.mark.parametrize("model", NATIVE_STRUCTURED_OUTPUT_MODELS_UNDER_TEST)
@pytest.mark.asyncio
async def test_native_structured_output_async(model):
    from crewai.llms.providers.anthropic.completion import (
        ANTHROPIC_STRUCTURED_OUTPUTS_BETA,
        AnthropicCompletion,
    )

    llm = AnthropicCompletion(model=model)
    mock_client = MagicMock()
    mock_client.beta.messages.create = AsyncMock(
        return_value=_structured_text_response()
    )
    llm._async_client = mock_client

    result = await llm.acall("What is the answer?", response_model=_Answer)

    assert result == _Answer(answer="42")
    kwargs = mock_client.beta.messages.create.call_args.kwargs
    assert ANTHROPIC_STRUCTURED_OUTPUTS_BETA in kwargs["betas"]
    assert kwargs["extra_body"]["output_format"]["type"] == "json_schema"


@pytest.mark.parametrize("model", NATIVE_STRUCTURED_OUTPUT_MODELS_UNDER_TEST)
def test_native_structured_output_sync_streaming(model):
    from crewai.llms.providers.anthropic.completion import (
        ANTHROPIC_STRUCTURED_OUTPUTS_BETA,
        AnthropicCompletion,
    )

    llm = AnthropicCompletion(model=model, stream=True)
    mock_client = MagicMock()
    mock_client.beta.messages.stream.return_value = _SyncAnthropicStream(
        _structured_stream_events(), _structured_text_response()
    )
    llm._client = mock_client
    llm._emit_stream_chunk_event = MagicMock()

    result = llm.call("What is the answer?", response_model=_Answer)

    assert result == _Answer(answer="42")
    kwargs = mock_client.beta.messages.stream.call_args.kwargs
    assert ANTHROPIC_STRUCTURED_OUTPUTS_BETA in kwargs["betas"]
    assert kwargs["extra_body"]["output_format"]["type"] == "json_schema"


@pytest.mark.parametrize("model", NATIVE_STRUCTURED_OUTPUT_MODELS_UNDER_TEST)
@pytest.mark.asyncio
async def test_native_structured_output_async_streaming(model):
    from crewai.llms.providers.anthropic.completion import (
        ANTHROPIC_STRUCTURED_OUTPUTS_BETA,
        AnthropicCompletion,
    )

    llm = AnthropicCompletion(model=model, stream=True)
    mock_client = MagicMock()
    mock_client.beta.messages.stream.return_value = _AsyncAnthropicStream(
        _structured_stream_events(), _structured_text_response()
    )
    llm._async_client = mock_client
    llm._emit_stream_chunk_event = MagicMock()

    result = await llm.acall("What is the answer?", response_model=_Answer)

    assert result == _Answer(answer="42")
    kwargs = mock_client.beta.messages.stream.call_args.kwargs
    assert ANTHROPIC_STRUCTURED_OUTPUTS_BETA in kwargs["betas"]
    assert kwargs["extra_body"]["output_format"]["type"] == "json_schema"


def test_native_structured_output_keeps_caller_tools():
    """The native path leaves the caller's tools in place; the fallback replaces them."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-opus-5")
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = _structured_text_response()
    llm._client = mock_client

    llm.call("What is the answer?", tools=[_WEATHER_TOOL], response_model=_Answer)

    sent_tools = mock_client.beta.messages.create.call_args.kwargs["tools"]
    assert [tool["name"] for tool in sent_tools] == ["get_weather"]


def test_tool_fallback_still_used_for_models_without_native_support():
    """Models outside the supported set keep the forced-tool-call behavior."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-3-5-haiku-20241022")
    mock_response = MagicMock()
    mock_response.content = [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "structured_output",
            "input": {"answer": "42"},
        }
    ]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = "tool_use"
    mock_response.id = "msg_fallback"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    llm._client = mock_client

    result = llm.call("What is the answer?", response_model=_Answer)

    assert result == _Answer(answer="42")
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "structured_output"}
    assert "betas" not in kwargs
    mock_client.beta.messages.create.assert_not_called()


def _final_answer_response(text: str):
    from anthropic.types import TextBlock

    mock_response = MagicMock()
    mock_response.content = [TextBlock(type="text", text=text, citations=None)]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = "end_turn"
    mock_response.id = "msg_forced_answer"
    return mock_response


@pytest.mark.parametrize(
    "history_tail",
    [
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "get_data", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "name": "get_data", "content": "partial"},
            {"role": "user", "content": "Analyze the tool result."},
        ],
        [
            {
                "role": "assistant",
                "content": "Thought: I need data\nAction: get_data\nAction Input: {}\nObservation: partial",
            },
        ],
    ],
    ids=["native-tools", "react"],
)
def test_max_iterations_request_ends_on_a_user_turn(history_tail):
    """Claude 4.6+ rejects a request whose last message is an assistant turn (no prefill).

    Regression for the max_iter path: the forced final-answer instruction must
    reach the Anthropic API as the trailing user message, on both loop shapes.
    """
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    from crewai.utilities.agent_utils import handle_max_iterations_exceeded
    from crewai.utilities.i18n import I18N_DEFAULT

    llm = AnthropicCompletion(model="claude-opus-5")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _final_answer_response("Final Answer: 42")
    llm._client = mock_client

    history = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Collect all the data."},
        *history_tail,
    ]

    result = handle_max_iterations_exceeded(
        printer=MagicMock(), messages=history, llm=llm, callbacks=[], verbose=False
    )

    mock_client.messages.create.assert_called_once()
    sent = mock_client.messages.create.call_args.kwargs["messages"]
    assert sent[-1] == {"role": "user", "content": I18N_DEFAULT.errors("force_final_answer")}
    assert result.output == "42"
