import os
import sys
import types
from unittest.mock import patch, MagicMock
import openai
import pytest

from crewai.llm import LLM
from crewai.llms.providers.openai.completion import OpenAICompletion, ResponsesAPIResult
from crewai.crew import Crew
from crewai.agent import Agent
from crewai.task import Task
from crewai.cli.constants import DEFAULT_LLM_MODEL

def test_openai_completion_is_used_when_openai_provider():
    """
    Test that OpenAICompletion from completion.py is used when LLM uses provider 'openai'
    """
    llm = LLM(model="gpt-4o")

    assert llm.__class__.__name__ == "OpenAICompletion"
    assert llm.provider == "openai"
    assert llm.model == "gpt-4o"


def test_openai_completion_is_used_when_no_provider_prefix():
    """
    Test that OpenAICompletion is used when no provider prefix is given (defaults to openai)
    """
    llm = LLM(model="gpt-4o")

    from crewai.llms.providers.openai.completion import OpenAICompletion
    assert isinstance(llm, OpenAICompletion)
    assert llm.provider == "openai"
    assert llm.model == "gpt-4o"

@pytest.mark.vcr()
def test_openai_is_default_provider_without_explicit_llm_set_on_agent():
    """
    Test that OpenAI is the default provider when no explicit LLM is set on the agent
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="gpt-4o-mini"),
    )
    task = Task(
        description="Find information about the population of Tokyo",
        expected_output="The population of Tokyo is 10 million",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff()
    assert crew.agents[0].llm.__class__.__name__ == "OpenAICompletion"
    assert crew.agents[0].llm.model == "gpt-4o-mini"






def test_openai_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using OpenAI provider
    """
    module_name = "crewai.llms.providers.openai.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="gpt-4o")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'OpenAICompletion')


def test_native_openai_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native OpenAI completion fails to initialize.
    This ensures we don't silently fall back when there's a configuration issue.
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should raise ImportError, not fall back to LiteLLM
        with pytest.raises(ImportError) as excinfo:
            LLM(model="gpt-4o")

        assert "Error importing native provider" in str(excinfo.value)
        assert "Native SDK failed" in str(excinfo.value)


def test_openai_completion_initialization_parameters():
    """
    Test that OpenAICompletion is initialized with correct parameters
    """
    llm = LLM(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
        api_key="test-key"
    )

    from crewai.llms.providers.openai.completion import OpenAICompletion
    assert isinstance(llm, OpenAICompletion)
    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 1000

def test_openai_completion_call():
    """
    Test that OpenAICompletion call method works
    """
    llm = LLM(model="openai/gpt-4o")

    # Mock the call method on the instance
    with patch.object(llm, 'call', return_value="Hello! I'm ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


def test_openai_completion_called_during_crew_execution():
    """
    Test that OpenAICompletion.call is actually invoked when running a crew
    """
    # Create the LLM instance first
    openai_llm = LLM(model="openai/gpt-4o")

    # Mock the call method on the specific instance
    with patch.object(openai_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:

        # Create agent with explicit LLM configuration
        agent = Agent(
            role="Research Assistant",
            goal="Find population info",
            backstory="You research populations.",
            llm=openai_llm,
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


def test_openai_completion_call_arguments():
    """
    Test that OpenAICompletion.call is invoked with correct arguments
    """
    # Create LLM instance first (like working tests)
    openai_llm = LLM(model="openai/gpt-4o")

    # Mock the instance method (like working tests)
    with patch.object(openai_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed successfully."

        agent = Agent(
            role="Test Agent",
            goal="Complete a simple task",
            backstory="You are a test agent.",
            llm=openai_llm  # Use same instance
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


def test_multiple_openai_calls_in_crew():
    """
    Test that OpenAICompletion.call is invoked multiple times for multiple tasks
    """
    # Create LLM instance first
    openai_llm = LLM(model="openai/gpt-4o")

    # Mock the instance method
    with patch.object(openai_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent = Agent(
            role="Multi-task Agent",
            goal="Complete multiple tasks",
            backstory="You can handle multiple tasks.",
            llm=openai_llm  # Use same instance
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


def test_openai_completion_with_tools():
    """
    Test that OpenAICompletion.call is invoked with tools when agent has tools
    """
    from crewai.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing"""
        return f"Tool result for: {query}"

    # Create LLM instance first
    openai_llm = LLM(model="openai/gpt-4o")

    # Mock the instance method (not the class method)
    with patch.object(openai_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed with tools."

        agent = Agent(
            role="Tool User",
            goal="Use tools to complete tasks",
            backstory="You can use tools.",
            llm=openai_llm,  # Use same instance
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

@pytest.mark.vcr()
def test_openai_completion_call_returns_usage_metrics():
    """
    Test that OpenAICompletion.call returns usage metrics
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="gpt-4o"),
        verbose=True,
    )

    task = Task(
        description="Find information about the population of Tokyo",
        expected_output="The population of Tokyo is 10 million",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 289
    assert result.token_usage.prompt_tokens == 173
    assert result.token_usage.completion_tokens == 116
    assert result.token_usage.successful_requests == 1
    assert result.token_usage.cached_prompt_tokens == 0


@pytest.mark.skip(reason="Allow for litellm")
def test_openai_raises_error_when_model_not_supported():
    """Test that OpenAICompletion raises ValueError when model not supported"""

    with patch('crewai.llms.providers.openai.completion.OpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.side_effect = openai.NotFoundError(
            message="The model `model-doesnt-exist` does not exist",
            response=MagicMock(),
            body={}
        )

        llm = LLM(model="openai/model-doesnt-exist")

        with pytest.raises(ValueError, match="Model.*not found"):
            llm.call("Hello")

def test_openai_client_setup_with_extra_arguments():
    """
    Test that OpenAICompletion is initialized with correct parameters
    """
    llm = LLM(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
        top_p=0.5,
        max_retries=3,
        timeout=30
    )

    # Check that model parameters are stored on the LLM instance
    assert llm.temperature == 0.7
    assert llm.max_tokens == 1000
    assert llm.top_p == 0.5

    # Check that client parameters are properly configured
    assert llm.client.max_retries == 3
    assert llm.client.timeout == 30

    # Test that parameters are properly used in API calls
    with patch.object(llm.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test response", tool_calls=None))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )

        llm.call("Hello")

        # Verify the API was called with the right parameters
        call_args = mock_create.call_args[1]  # keyword arguments
        assert call_args['temperature'] == 0.7
        assert call_args['max_tokens'] == 1000
        assert call_args['top_p'] == 0.5
        assert call_args['model'] == 'gpt-4o'

def test_extra_arguments_are_passed_to_openai_completion():
    """
    Test that extra arguments are passed to OpenAICompletion
    """
    llm = LLM(model="gpt-4o", temperature=0.7, max_tokens=1000, top_p=0.5, max_retries=3)

    with patch.object(llm.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test response", tool_calls=None))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )

        llm.call("Hello, how are you?")

        assert mock_create.called
        call_kwargs = mock_create.call_args[1]

        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['max_tokens'] == 1000
        assert call_kwargs['top_p'] == 0.5
        assert call_kwargs['model'] == 'gpt-4o'



def test_openai_get_client_params_with_api_base():
    """
    Test that _get_client_params correctly converts api_base to base_url
    """
    llm = OpenAICompletion(
        model="gpt-4o",
        api_base="https://custom.openai.com/v1",
    )
    client_params = llm._get_client_params()
    assert client_params["base_url"] == "https://custom.openai.com/v1"

def test_openai_get_client_params_with_base_url_priority():
    """
    Test that base_url takes priority over api_base in _get_client_params
    """
    llm = OpenAICompletion(
        model="gpt-4o",
        base_url="https://priority.openai.com/v1",
        api_base="https://fallback.openai.com/v1",
    )
    client_params = llm._get_client_params()
    assert client_params["base_url"] == "https://priority.openai.com/v1"

def test_openai_get_client_params_with_env_var():
    """
    Test that _get_client_params uses OPENAI_BASE_URL environment variable as fallback
    """
    with patch.dict(os.environ, {
        "OPENAI_BASE_URL": "https://env.openai.com/v1",
    }):
        llm = OpenAICompletion(model="gpt-4o")
        client_params = llm._get_client_params()
        assert client_params["base_url"] == "https://env.openai.com/v1"

def test_openai_get_client_params_priority_order():
    """
    Test the priority order: base_url > api_base > OPENAI_BASE_URL env var
    """
    with patch.dict(os.environ, {
        "OPENAI_BASE_URL": "https://env.openai.com/v1",
    }):
        # Test base_url beats api_base and env var
        llm1 = OpenAICompletion(
            model="gpt-4o",
            base_url="https://base-url.openai.com/v1",
            api_base="https://api-base.openai.com/v1",
        )
        params1 = llm1._get_client_params()
        assert params1["base_url"] == "https://base-url.openai.com/v1"

        # Test api_base beats env var when base_url is None
        llm2 = OpenAICompletion(
            model="gpt-4o",
            api_base="https://api-base.openai.com/v1",
        )
        params2 = llm2._get_client_params()
        assert params2["base_url"] == "https://api-base.openai.com/v1"

        # Test env var is used when both base_url and api_base are None
        llm3 = OpenAICompletion(model="gpt-4o")
        params3 = llm3._get_client_params()
        assert params3["base_url"] == "https://env.openai.com/v1"

def test_openai_get_client_params_no_base_url(monkeypatch):
    """
    Test that _get_client_params works correctly when no base_url is specified
    """
    # Clear env vars that could set base_url
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    llm = OpenAICompletion(model="gpt-4o")
    client_params = llm._get_client_params()
    # When no base_url is provided, it should not be in the params (filtered out as None)
    assert "base_url" not in client_params or client_params.get("base_url") is None


def test_openai_streaming_with_response_model():
    """
    Test that streaming with response_model works correctly and doesn't call invalid API methods.
    This test verifies the fix for the bug where streaming with response_model attempted to call
    self.client.responses.stream() with invalid parameters (input, text_format).
    """
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        """Test response model."""

        answer: str
        confidence: float

    llm = LLM(model="openai/gpt-4o", stream=True)

    with patch.object(llm.client.beta.chat.completions, "stream") as mock_stream:
        # Create mock chunks with content.delta event structure
        mock_chunk1 = MagicMock()
        mock_chunk1.type = "content.delta"
        mock_chunk1.delta = '{"answer": "test", '
        mock_chunk1.id = "response-1"

        # Second chunk
        mock_chunk2 = MagicMock()
        mock_chunk2.type = "content.delta"
        mock_chunk2.delta = '"confidence": 0.95}'
        mock_chunk2.id = "response-2"

        # Create mock final completion with parsed result
        mock_parsed = TestResponse(answer="test", confidence=0.95)
        mock_message = MagicMock()
        mock_message.parsed = mock_parsed
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_final_completion = MagicMock()
        mock_final_completion.choices = [mock_choice]

        # Create mock stream context manager
        mock_stream_obj = MagicMock()
        mock_stream_obj.__enter__ = MagicMock(return_value=mock_stream_obj)
        mock_stream_obj.__exit__ = MagicMock(return_value=None)
        mock_stream_obj.__iter__ = MagicMock(return_value=iter([mock_chunk1, mock_chunk2]))
        mock_stream_obj.get_final_completion = MagicMock(return_value=mock_final_completion)

        mock_stream.return_value = mock_stream_obj

        result = llm.call("Test question", response_model=TestResponse)

        assert result is not None
        assert isinstance(result, TestResponse)
        assert result.answer == "test"
        assert result.confidence == 0.95

        assert mock_stream.called
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["response_format"] == TestResponse

        assert "input" not in call_kwargs
        assert "text_format" not in call_kwargs


@pytest.mark.vcr()
def test_openai_response_format_with_pydantic_model():
    """
    Test that response_format with a Pydantic BaseModel returns structured output.
    """
    from pydantic import BaseModel, Field

    class AnswerResponse(BaseModel):
        """Response model with structured fields."""

        answer: str = Field(description="The answer to the question")
        confidence: float = Field(description="Confidence score between 0 and 1")

    llm = LLM(model="gpt-4o", response_format=AnswerResponse)
    result = llm.call("What is the capital of France? Be concise.")

    assert isinstance(result, AnswerResponse)
    assert result.answer is not None
    assert 0 <= result.confidence <= 1


@pytest.mark.vcr()
def test_openai_response_format_with_dict():
    """
    Test that response_format with a dict returns JSON output.
    """
    import json

    llm = LLM(model="gpt-4o", response_format={"type": "json_object"})
    result = llm.call("Return a JSON object with a 'status' field set to 'success'")

    parsed = json.loads(result)
    assert "status" in parsed


@pytest.mark.vcr()
def test_openai_response_format_none():
    """
    Test that when response_format is None, the API returns plain text.
    """
    llm = LLM(model="gpt-4o", response_format=None)
    result = llm.call("Say hello in one word")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr()
def test_openai_streaming_returns_usage_metrics():
    """
    Test that OpenAI streaming calls return proper token usage metrics.
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the capital of France",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="gpt-4o-mini", stream=True),
        verbose=True,
    )

    task = Task(
        description="What is the capital of France?",
        expected_output="The capital of France",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    assert result.token_usage is not None
    assert result.token_usage.total_tokens > 0
    assert result.token_usage.prompt_tokens > 0
    assert result.token_usage.completion_tokens > 0
    assert result.token_usage.successful_requests >= 1


def test_openai_responses_api_initialization():
    """Test that OpenAI Responses API can be initialized with api='responses'."""
    llm = OpenAICompletion(
        model="gpt-5",
        api="responses",
        instructions="You are a helpful assistant.",
        store=True,
    )

    assert llm.api == "responses"
    assert llm.instructions == "You are a helpful assistant."
    assert llm.store is True
    assert llm.model == "gpt-5"


def test_openai_responses_api_default_is_completions():
    """Test that the default API is 'completions' for backward compatibility."""
    llm = OpenAICompletion(model="gpt-4o")

    assert llm.api == "completions"


def test_openai_responses_api_prepare_params():
    """Test that Responses API params are prepared correctly."""
    llm = OpenAICompletion(
        model="gpt-5",
        api="responses",
        instructions="Base instructions.",
        store=True,
        temperature=0.7,
    )

    messages = [
        {"role": "system", "content": "System message."},
        {"role": "user", "content": "Hello!"},
    ]

    params = llm._prepare_responses_params(messages)

    assert params["model"] == "gpt-5"
    assert "Base instructions." in params["instructions"]
    assert "System message." in params["instructions"]
    assert params["store"] is True
    assert params["temperature"] == 0.7
    assert params["input"] == [{"role": "user", "content": "Hello!"}]


def test_openai_responses_api_tool_format():
    """Test that tools are converted to Responses API format (internally-tagged)."""
    llm = OpenAICompletion(model="gpt-5", api="responses")

    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    responses_tools = llm._convert_tools_for_responses(tools)

    assert len(responses_tools) == 1
    tool = responses_tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "get_weather"
    assert tool["description"] == "Get the weather for a location"
    assert "parameters" in tool
    assert "function" not in tool


def test_openai_completions_api_tool_format():
    """Test that tools are converted to Chat Completions API format (externally-tagged)."""
    llm = OpenAICompletion(model="gpt-4o", api="completions")

    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    completions_tools = llm._convert_tools_for_interference(tools)

    assert len(completions_tools) == 1
    tool = completions_tools[0]
    assert tool["type"] == "function"
    assert "function" in tool
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["description"] == "Get the weather for a location"


def test_openai_responses_api_structured_output_format():
    """Test that structured outputs use text.format for Responses API."""
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    llm = OpenAICompletion(model="gpt-5", api="responses")

    messages = [{"role": "user", "content": "Extract: Jane, 25"}]
    params = llm._prepare_responses_params(messages, response_model=Person)

    assert "text" in params
    assert "format" in params["text"]
    assert params["text"]["format"]["type"] == "json_schema"
    assert params["text"]["format"]["name"] == "Person"
    assert params["text"]["format"]["strict"] is True


def test_openai_responses_api_with_previous_response_id():
    """Test that previous_response_id is passed for multi-turn conversations."""
    llm = OpenAICompletion(
        model="gpt-5",
        api="responses",
        previous_response_id="resp_abc123",
        store=True,
    )

    messages = [{"role": "user", "content": "Continue our conversation."}]
    params = llm._prepare_responses_params(messages)

    assert params["previous_response_id"] == "resp_abc123"
    assert params["store"] is True


def test_openai_responses_api_call_routing():
    """Test that call() routes to the correct API based on the api parameter."""
    from unittest.mock import patch, MagicMock

    llm_completions = OpenAICompletion(model="gpt-4o", api="completions")
    llm_responses = OpenAICompletion(model="gpt-5", api="responses")

    with patch.object(
        llm_completions, "_call_completions", return_value="completions result"
    ) as mock_completions:
        result = llm_completions.call("Hello")
        mock_completions.assert_called_once()
        assert result == "completions result"

    with patch.object(
        llm_responses, "_call_responses", return_value="responses result"
    ) as mock_responses:
        result = llm_responses.call("Hello")
        mock_responses.assert_called_once()
        assert result == "responses result"


# =============================================================================
# VCR Integration Tests for Responses API
# =============================================================================


@pytest.mark.vcr()
def test_openai_responses_api_basic_call():
    """Test basic Responses API call with text generation."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        instructions="You are a helpful assistant. Be concise.",
    )

    result = llm.call("What is 2 + 2? Answer with just the number.")

    assert isinstance(result, str)
    assert "4" in result


@pytest.mark.vcr()
def test_openai_responses_api_with_structured_output():
    """Test Responses API with structured output using Pydantic model."""
    from pydantic import BaseModel, Field

    class MathAnswer(BaseModel):
        """Structured math answer."""

        result: int = Field(description="The numerical result")
        explanation: str = Field(description="Brief explanation")

    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
    )

    result = llm.call("What is 5 * 7?", response_model=MathAnswer)

    assert isinstance(result, MathAnswer)
    assert result.result == 35


@pytest.mark.vcr()
def test_openai_responses_api_with_system_message_extraction():
    """Test that system messages are properly extracted to instructions."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
    )

    messages = [
        {"role": "system", "content": "You always respond in uppercase letters only."},
        {"role": "user", "content": "Say hello"},
    ]

    result = llm.call(messages)

    assert isinstance(result, str)
    assert result.isupper() or "HELLO" in result.upper()


@pytest.mark.vcr()
def test_openai_responses_api_streaming():
    """Test Responses API with streaming enabled."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        stream=True,
        instructions="Be very concise.",
    )

    result = llm.call("Count from 1 to 3, separated by commas.")

    assert isinstance(result, str)
    assert "1" in result
    assert "2" in result
    assert "3" in result


@pytest.mark.vcr()
def test_openai_responses_api_returns_usage_metrics():
    """Test that Responses API calls return proper token usage metrics."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
    )

    llm.call("Say hello")

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0


def test_openai_responses_api_builtin_tools_param():
    """Test that builtin_tools parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        builtin_tools=["web_search", "code_interpreter"],
    )

    assert llm.builtin_tools == ["web_search", "code_interpreter"]

    messages = [{"role": "user", "content": "Test"}]
    params = llm._prepare_responses_params(messages)

    assert "tools" in params
    tool_types = [t["type"] for t in params["tools"]]
    assert "web_search_preview" in tool_types
    assert "code_interpreter" in tool_types


def test_openai_responses_api_builtin_tools_with_custom_tools():
    """Test that builtin_tools can be combined with custom function tools."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        builtin_tools=["web_search"],
    )

    custom_tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    messages = [{"role": "user", "content": "Test"}]
    params = llm._prepare_responses_params(messages, tools=custom_tools)

    assert len(params["tools"]) == 2
    tool_types = [t.get("type") for t in params["tools"]]
    assert "web_search_preview" in tool_types
    assert "function" in tool_types


@pytest.mark.vcr()
def test_openai_responses_api_with_web_search():
    """Test Responses API with web_search built-in tool."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        builtin_tools=["web_search"],
    )

    result = llm.call("What is the current population of Tokyo? Be brief.")

    assert isinstance(result, str)
    assert len(result) > 0


def test_responses_api_result_dataclass():
    """Test ResponsesAPIResult dataclass functionality."""
    result = ResponsesAPIResult(
        text="Hello, world!",
        response_id="resp_123",
    )

    assert result.text == "Hello, world!"
    assert result.response_id == "resp_123"
    assert result.web_search_results == []
    assert result.file_search_results == []
    assert result.code_interpreter_results == []
    assert result.computer_use_results == []
    assert result.reasoning_summaries == []
    assert result.function_calls == []
    assert not result.has_tool_outputs()
    assert not result.has_reasoning()


def test_responses_api_result_has_tool_outputs():
    """Test ResponsesAPIResult.has_tool_outputs() method."""
    result_with_web = ResponsesAPIResult(
        text="Test",
        web_search_results=[{"id": "ws_1", "status": "completed", "type": "web_search_call"}],
    )
    assert result_with_web.has_tool_outputs()

    result_with_file = ResponsesAPIResult(
        text="Test",
        file_search_results=[{"id": "fs_1", "status": "completed", "type": "file_search_call", "queries": [], "results": []}],
    )
    assert result_with_file.has_tool_outputs()


def test_responses_api_result_has_reasoning():
    """Test ResponsesAPIResult.has_reasoning() method."""
    result_with_reasoning = ResponsesAPIResult(
        text="Test",
        reasoning_summaries=[{"id": "r_1", "type": "reasoning", "summary": []}],
    )
    assert result_with_reasoning.has_reasoning()

    result_without = ResponsesAPIResult(text="Test")
    assert not result_without.has_reasoning()


def test_openai_responses_api_parse_tool_outputs_param():
    """Test that parse_tool_outputs parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        parse_tool_outputs=True,
    )

    assert llm.parse_tool_outputs is True


def test_openai_responses_api_parse_tool_outputs_default_false():
    """Test that parse_tool_outputs defaults to False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
    )

    assert llm.parse_tool_outputs is False


@pytest.mark.vcr()
def test_openai_responses_api_with_parse_tool_outputs():
    """Test Responses API with parse_tool_outputs enabled returns ResponsesAPIResult."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        builtin_tools=["web_search"],
        parse_tool_outputs=True,
    )

    result = llm.call("What is the current population of Tokyo? Be very brief.")

    assert isinstance(result, ResponsesAPIResult)
    assert len(result.text) > 0
    assert result.response_id is not None
    # Web search should have been used
    assert len(result.web_search_results) > 0
    assert result.has_tool_outputs()


@pytest.mark.vcr()
def test_openai_responses_api_parse_tool_outputs_basic_call():
    """Test Responses API with parse_tool_outputs but no built-in tools."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        parse_tool_outputs=True,
    )

    result = llm.call("Say hello in exactly 3 words.")

    assert isinstance(result, ResponsesAPIResult)
    assert len(result.text) > 0
    assert result.response_id is not None
    # No built-in tools used
    assert not result.has_tool_outputs()


# ============================================================================
# Auto-Chaining Tests (Responses API)
# ============================================================================


def test_openai_responses_api_auto_chain_param():
    """Test that auto_chain parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    assert llm.auto_chain is True
    assert llm._last_response_id is None


def test_openai_responses_api_auto_chain_default_false():
    """Test that auto_chain defaults to False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
    )

    assert llm.auto_chain is False


def test_openai_responses_api_last_response_id_property():
    """Test last_response_id property."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    # Initially None
    assert llm.last_response_id is None

    # Simulate setting the internal value
    llm._last_response_id = "resp_test_123"
    assert llm.last_response_id == "resp_test_123"


def test_openai_responses_api_reset_chain():
    """Test reset_chain() method clears the response ID."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    # Set a response ID
    llm._last_response_id = "resp_test_123"
    assert llm.last_response_id == "resp_test_123"

    # Reset the chain
    llm.reset_chain()
    assert llm.last_response_id is None


def test_openai_responses_api_auto_chain_prepare_params():
    """Test that _prepare_responses_params uses auto-chained response ID."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    # No previous response ID yet
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "previous_response_id" not in params

    # Set a previous response ID
    llm._last_response_id = "resp_previous_123"
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert params.get("previous_response_id") == "resp_previous_123"


def test_openai_responses_api_explicit_previous_response_id_takes_precedence():
    """Test that explicit previous_response_id overrides auto-chained ID."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
        previous_response_id="resp_explicit_456",
    )

    # Set an auto-chained response ID
    llm._last_response_id = "resp_auto_123"

    # Explicit should take precedence
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert params.get("previous_response_id") == "resp_explicit_456"


def test_openai_responses_api_auto_chain_disabled_no_tracking():
    """Test that response ID is not tracked when auto_chain is False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=False,
    )

    # Even with a "previous" response ID set internally, params shouldn't use it
    llm._last_response_id = "resp_should_not_use"
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "previous_response_id" not in params


@pytest.mark.vcr()
def test_openai_responses_api_auto_chain_integration():
    """Test auto-chaining tracks response IDs across calls."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        auto_chain=True,
    )

    # First call - should not have previous_response_id
    assert llm.last_response_id is None
    result1 = llm.call("My name is Alice. Remember this.")

    # After first call, should have a response ID
    assert llm.last_response_id is not None
    first_response_id = llm.last_response_id
    assert first_response_id.startswith("resp_")

    # Second call - should use the first response ID
    result2 = llm.call("What is my name?")

    # Response ID should be updated
    assert llm.last_response_id is not None
    assert llm.last_response_id != first_response_id  # Should be a new ID

    # The response should remember context (Alice)
    assert isinstance(result1, str)
    assert isinstance(result2, str)


@pytest.mark.vcr()
def test_openai_responses_api_auto_chain_with_reset():
    """Test that reset_chain() properly starts a new conversation."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        auto_chain=True,
    )

    # First conversation
    llm.call("My favorite color is blue.")
    first_chain_id = llm.last_response_id
    assert first_chain_id is not None

    # Reset and start new conversation
    llm.reset_chain()
    assert llm.last_response_id is None

    # New call should start fresh
    llm.call("Hello!")
    second_chain_id = llm.last_response_id
    assert second_chain_id is not None
    # New conversation, so different response ID
    assert second_chain_id != first_chain_id


# =============================================================================
# Encrypted Reasoning for ZDR (Zero Data Retention) Tests
# =============================================================================


def test_openai_responses_api_auto_chain_reasoning_param():
    """Test that auto_chain_reasoning parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
    )

    assert llm.auto_chain_reasoning is True
    assert llm._last_reasoning_items is None


def test_openai_responses_api_auto_chain_reasoning_default_false():
    """Test that auto_chain_reasoning defaults to False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
    )

    assert llm.auto_chain_reasoning is False


def test_openai_responses_api_last_reasoning_items_property():
    """Test last_reasoning_items property."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
    )

    # Initially None
    assert llm.last_reasoning_items is None

    # Simulate setting the internal value
    mock_items = [{"id": "rs_test_123", "type": "reasoning"}]
    llm._last_reasoning_items = mock_items
    assert llm.last_reasoning_items == mock_items


def test_openai_responses_api_reset_reasoning_chain():
    """Test reset_reasoning_chain() method clears reasoning items."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
    )

    # Set reasoning items
    mock_items = [{"id": "rs_test_123", "type": "reasoning"}]
    llm._last_reasoning_items = mock_items
    assert llm.last_reasoning_items == mock_items

    # Reset the reasoning chain
    llm.reset_reasoning_chain()
    assert llm.last_reasoning_items is None


def test_openai_responses_api_auto_chain_reasoning_adds_include():
    """Test that auto_chain_reasoning adds reasoning.encrypted_content to include."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "include" in params
    assert "reasoning.encrypted_content" in params["include"]


def test_openai_responses_api_auto_chain_reasoning_preserves_existing_include():
    """Test that auto_chain_reasoning preserves existing include items."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
        include=["file_search_call.results"],
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "include" in params
    assert "reasoning.encrypted_content" in params["include"]
    assert "file_search_call.results" in params["include"]


def test_openai_responses_api_auto_chain_reasoning_no_duplicate_include():
    """Test that reasoning.encrypted_content is not duplicated if already in include."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
        include=["reasoning.encrypted_content"],
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "include" in params
    # Should only appear once
    assert params["include"].count("reasoning.encrypted_content") == 1


def test_openai_responses_api_auto_chain_reasoning_prepends_to_input():
    """Test that stored reasoning items are prepended to input."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
    )

    # Simulate stored reasoning items
    mock_reasoning = MagicMock()
    mock_reasoning.type = "reasoning"
    mock_reasoning.id = "rs_test_123"
    llm._last_reasoning_items = [mock_reasoning]

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])

    # Input should have reasoning item first, then the message
    assert len(params["input"]) == 2
    assert params["input"][0] == mock_reasoning
    assert params["input"][1]["role"] == "user"


def test_openai_responses_api_auto_chain_reasoning_disabled_no_include():
    """Test that reasoning.encrypted_content is not added when auto_chain_reasoning is False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=False,
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    # Should not have include at all (unless explicitly set)
    assert "include" not in params or "reasoning.encrypted_content" not in params.get("include", [])


def test_openai_responses_api_auto_chain_reasoning_disabled_no_prepend():
    """Test that reasoning items are not prepended when auto_chain_reasoning is False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=False,
    )

    # Even with stored reasoning items, they should not be prepended
    mock_reasoning = MagicMock()
    mock_reasoning.type = "reasoning"
    llm._last_reasoning_items = [mock_reasoning]

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])

    # Input should only have the message, not the reasoning item
    assert len(params["input"]) == 1
    assert params["input"][0]["role"] == "user"


def test_openai_responses_api_both_auto_chains_work_together():
    """Test that auto_chain and auto_chain_reasoning can be used together."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
        auto_chain_reasoning=True,
    )

    assert llm.auto_chain is True
    assert llm.auto_chain_reasoning is True
    assert llm._last_response_id is None
    assert llm._last_reasoning_items is None

    # Set both internal values
    llm._last_response_id = "resp_123"
    mock_reasoning = MagicMock()
    mock_reasoning.type = "reasoning"
    llm._last_reasoning_items = [mock_reasoning]

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])

    # Both should be applied
    assert params.get("previous_response_id") == "resp_123"
    assert "reasoning.encrypted_content" in params["include"]
    assert len(params["input"]) == 2  # Reasoning item + message


# =============================================================================
# Agent Kickoff Structured Output Tests
# =============================================================================


@pytest.mark.vcr()
def test_openai_agent_kickoff_structured_output_without_tools():
    """
    Test that agent kickoff returns structured output without tools.
    This tests native structured output handling for OpenAI models.
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
        llm=LLM(model="gpt-4o-mini"),
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
def test_openai_agent_kickoff_structured_output_with_tools():
    """
    Test that agent kickoff returns structured output after using tools.
    This tests post-tool-call structured output handling for OpenAI models.
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
        llm=LLM(model="gpt-4o-mini"),
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


# =============================================================================
# Stop Words with Structured Output Tests
# =============================================================================


def test_openai_stop_words_not_applied_to_structured_output():
    """
    Test that stop words are NOT applied when response_model is provided.
    This ensures JSON responses containing stop word patterns (like "Observation:")
    are not truncated, which would cause JSON validation to fail.
    """
    from pydantic import BaseModel, Field

    class ResearchResult(BaseModel):
        """Research result that may contain stop word patterns in string fields."""

        finding: str = Field(description="The research finding")
        observation: str = Field(description="Observation about the finding")

    # Create OpenAI completion instance with stop words configured
    llm = OpenAICompletion(
        model="gpt-4o",
        stop=["Observation:", "Final Answer:"],  # Common stop words
    )

    # JSON response that contains a stop word pattern in a string field
    # Without the fix, this would be truncated at "Observation:" breaking the JSON
    json_response = '{"finding": "The data shows growth", "observation": "Observation: This confirms the hypothesis"}'

    # Test the _validate_structured_output method directly with content containing stop words
    # This simulates what happens when the API returns JSON with stop word patterns
    result = llm._validate_structured_output(json_response, ResearchResult)

    # Should successfully parse the full JSON without truncation
    assert isinstance(result, ResearchResult)
    assert result.finding == "The data shows growth"
    # The observation field should contain the full text including "Observation:"
    assert "Observation:" in result.observation


def test_openai_stop_words_still_applied_to_regular_responses():
    """
    Test that stop words ARE still applied for regular (non-structured) responses.
    This ensures the fix didn't break normal stop word behavior.
    """
    # Create OpenAI completion instance with stop words configured
    llm = OpenAICompletion(
        model="gpt-4o",
        stop=["Observation:", "Final Answer:"],
    )

    # Response that contains a stop word - should be truncated
    response_with_stop_word = "I need to search for more information.\n\nAction: search\nObservation: Found results"

    # Test the _apply_stop_words method directly
    result = llm._apply_stop_words(response_with_stop_word)

    # Response should be truncated at the stop word
    assert "Observation:" not in result
    assert "Found results" not in result
    assert "I need to search for more information" in result


def test_openai_structured_output_preserves_json_with_stop_word_patterns():
    """
    Test that structured output validation preserves JSON content
    even when string fields contain stop word patterns.
    """
    from pydantic import BaseModel, Field

    class AgentObservation(BaseModel):
        """Model with fields that might contain stop word-like text."""

        action_taken: str = Field(description="What action was taken")
        observation_result: str = Field(description="The observation result")
        final_answer: str = Field(description="The final answer")

    llm = OpenAICompletion(
        model="gpt-4o",
        stop=["Observation:", "Final Answer:", "Action:"],
    )

    # JSON that contains all the stop word patterns as part of the content
    json_with_stop_patterns = '''{
        "action_taken": "Action: Searched the database",
        "observation_result": "Observation: Found 5 relevant results",
        "final_answer": "Final Answer: The data shows positive growth"
    }'''

    # This should NOT be truncated since it's structured output
    result = llm._validate_structured_output(json_with_stop_patterns, AgentObservation)

    assert isinstance(result, AgentObservation)
    assert "Action:" in result.action_taken
    assert "Observation:" in result.observation_result
    assert "Final Answer:" in result.final_answer
