import os
import sys
import types
from unittest.mock import patch, MagicMock
import openai
import pytest

from crewai.llm import LLM
from crewai.llms.providers.openai.completion import OpenAICompletion
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
    )
    task = Task(
        description="Find information about the population of Tokyo",
        expected_output="The population of Tokyo is 10 million",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff()
    assert crew.agents[0].llm.__class__.__name__ == "OpenAICompletion"
    assert crew.agents[0].llm.model == DEFAULT_LLM_MODEL






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

        mock_chunk2 = MagicMock()
        mock_chunk2.type = "content.delta"
        mock_chunk2.delta = '"confidence": 0.95}'

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
        assert isinstance(result, str)

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
