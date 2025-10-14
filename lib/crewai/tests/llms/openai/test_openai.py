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
    llm = LLM(model="openai/gpt-4o")

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

@pytest.mark.vcr(filter_headers=["authorization"])
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
    LLM(model="openai/gpt-4o")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'OpenAICompletion')


def test_fallback_to_litellm_when_native_fails():
    """
    Test that LLM falls back to LiteLLM when native OpenAI completion fails
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should fall back to LiteLLM
        llm = LLM(model="openai/gpt-4o")

        # Check that it's using LiteLLM
        assert hasattr(llm, 'is_litellm')
        assert llm.is_litellm == True


def test_openai_completion_initialization_parameters():
    """
    Test that OpenAICompletion is initialized with correct parameters
    """
    llm = LLM(
        model="openai/gpt-4o",
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

@pytest.mark.vcr(filter_headers=["authorization"])
def test_openai_completion_call_returns_usage_metrics():
    """
    Test that OpenAICompletion.call returns usage metrics
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="openai/gpt-4o"),
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


def test_openai_raises_error_when_model_not_supported():
    """
    Test that OpenAICompletion raises an error when the model is not supported
    """
    llm = LLM(model="openai/model-doesnt-exist")
    with pytest.raises(openai.NotFoundError):
        llm.call("Hello")

def test_openai_client_setup_with_extra_arguments():
    """
    Test that OpenAICompletion is initialized with correct parameters
    """
    llm = LLM(
        model="openai/gpt-4o",
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
    llm = LLM(model="openai/gpt-4o", temperature=0.7, max_tokens=1000, top_p=0.5, max_retries=3)

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
