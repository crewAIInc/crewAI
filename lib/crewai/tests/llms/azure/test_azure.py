import os
import sys
import types
from unittest.mock import patch, MagicMock, Mock
import pytest

from crewai.llm import LLM
from crewai.crew import Crew
from crewai.agent import Agent
from crewai.task import Task


@pytest.fixture
def mock_azure_credentials():
    """Mock Azure credentials for tests that need them."""
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com"
    }):
        yield


@pytest.mark.usefixtures("mock_azure_credentials")
def test_azure_completion_is_used_when_azure_provider():
    """
    Test that AzureCompletion from completion.py is used when LLM uses provider 'azure'
    """
    llm = LLM(model="azure/gpt-4")

    assert llm.__class__.__name__ == "AzureCompletion"
    assert llm.provider == "azure"
    assert llm.model == "gpt-4"


@pytest.mark.usefixtures("mock_azure_credentials")
def test_azure_completion_is_used_when_azure_openai_provider():
    """
    Test that AzureCompletion is used when provider is 'azure_openai'
    """
    llm = LLM(model="azure_openai/gpt-4")

    from crewai.llms.providers.azure.completion import AzureCompletion
    assert isinstance(llm, AzureCompletion)
    assert llm.provider == "azure"
    assert llm.model == "gpt-4"


def test_azure_tool_use_conversation_flow():
    """
    Test that the Azure completion properly handles tool use conversation flow
    """
    from crewai.llms.providers.azure.completion import AzureCompletion
    from azure.ai.inference.models import ChatCompletionsToolCall

    # Create AzureCompletion instance
    completion = AzureCompletion(
        model="gpt-4",
        api_key="test-key",
        endpoint="https://test.openai.azure.com"
    )

    # Mock tool function
    def mock_weather_tool(location: str) -> str:
        return f"The weather in {location} is sunny and 75°F"

    available_functions = {"get_weather": mock_weather_tool}

    # Mock the Azure client responses
    with patch.object(completion.client, 'complete') as mock_complete:
        # Mock tool call in response with proper type
        mock_tool_call = MagicMock(spec=ChatCompletionsToolCall)
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "San Francisco"}'

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        mock_complete.return_value = mock_response

        # Test the call
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        result = completion.call(
            messages=messages,
            available_functions=available_functions
        )

        # Verify the tool was executed and returned the result
        assert result == "The weather in San Francisco is sunny and 75°F"

        # Verify that the API was called
        assert mock_complete.called

@pytest.mark.usefixtures("mock_azure_credentials")
def test_azure_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using Azure provider
    """
    module_name = "crewai.llms.providers.azure.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="azure/gpt-4")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'AzureCompletion')


def test_native_azure_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native Azure completion fails to initialize.
    This ensures we don't silently fall back when there's a configuration issue.
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native Azure AI Inference SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should raise ImportError, not fall back to LiteLLM
        with pytest.raises(ImportError) as excinfo:
            LLM(model="azure/gpt-4")

        assert "Error importing native provider" in str(excinfo.value)
        assert "Native Azure AI Inference SDK failed" in str(excinfo.value)


def test_azure_completion_initialization_parameters():
    """
    Test that AzureCompletion is initialized with correct parameters
    """
    llm = LLM(
        model="azure/gpt-4",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        api_key="test-key",
        endpoint="https://test.openai.azure.com"
    )

    from crewai.llms.providers.azure.completion import AzureCompletion
    assert isinstance(llm, AzureCompletion)
    assert llm.model == "gpt-4"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2000
    assert llm.top_p == 0.9
    assert llm.frequency_penalty == 0.5
    assert llm.presence_penalty == 0.3


def test_azure_specific_parameters():
    """
    Test Azure-specific parameters like stop sequences, streaming, and API version
    """
    llm = LLM(
        model="azure/gpt-4",
        stop=["Human:", "Assistant:"],
        stream=True,
        api_version="2024-02-01",
        endpoint="https://test.openai.azure.com"
    )

    from crewai.llms.providers.azure.completion import AzureCompletion
    assert isinstance(llm, AzureCompletion)
    assert llm.stop == ["Human:", "Assistant:"]
    assert llm.stream == True
    assert llm.api_version == "2024-02-01"


@pytest.mark.usefixtures("mock_azure_credentials")
def test_azure_completion_call():
    """
    Test that AzureCompletion call method works
    """
    llm = LLM(model="azure/gpt-4")

    # Mock the call method on the instance
    with patch.object(llm, 'call', return_value="Hello! I'm Azure OpenAI, ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm Azure OpenAI, ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


@pytest.mark.usefixtures("mock_azure_credentials")
def test_azure_completion_called_during_crew_execution():
    """
    Test that AzureCompletion.call is actually invoked when running a crew
    """
    # Create the LLM instance first
    azure_llm = LLM(model="azure/gpt-4")

    # Mock the call method on the specific instance
    with patch.object(azure_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:

        # Create agent with explicit LLM configuration
        agent = Agent(
            role="Research Assistant",
            goal="Find population info",
            backstory="You research populations.",
            llm=azure_llm,
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


@pytest.mark.usefixtures("mock_azure_credentials")
def test_azure_completion_call_arguments():
    """
    Test that AzureCompletion.call is invoked with correct arguments
    """
    # Create LLM instance first
    azure_llm = LLM(model="azure/gpt-4")

    # Mock the instance method
    with patch.object(azure_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed successfully."

        agent = Agent(
            role="Test Agent",
            goal="Complete a simple task",
            backstory="You are a test agent.",
            llm=azure_llm  # Use same instance
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


def test_multiple_azure_calls_in_crew():
    """
    Test that AzureCompletion.call is invoked multiple times for multiple tasks
    """
    # Create LLM instance first
    azure_llm = LLM(model="azure/gpt-4")

    # Mock the instance method
    with patch.object(azure_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent = Agent(
            role="Multi-task Agent",
            goal="Complete multiple tasks",
            backstory="You can handle multiple tasks.",
            llm=azure_llm  # Use same instance
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


def test_azure_completion_with_tools():
    """
    Test that AzureCompletion.call is invoked with tools when agent has tools
    """
    from crewai.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing"""
        return f"Tool result for: {query}"

    # Create LLM instance first
    azure_llm = LLM(model="azure/gpt-4")

    # Mock the instance method
    with patch.object(azure_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed with tools."

        agent = Agent(
            role="Tool User",
            goal="Use tools to complete tasks",
            backstory="You can use tools.",
            llm=azure_llm,  # Use same instance
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


def test_azure_raises_error_when_endpoint_missing():
    """Test that AzureCompletion raises ValueError when endpoint is missing"""
    from crewai.llms.providers.azure.completion import AzureCompletion

    # Clear environment variables
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Azure endpoint is required"):
            AzureCompletion(model="gpt-4", api_key="test-key")


def test_azure_raises_error_when_api_key_missing():
    """Test that AzureCompletion raises ValueError when API key is missing"""
    from crewai.llms.providers.azure.completion import AzureCompletion

    # Clear environment variables
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Azure API key is required"):
            AzureCompletion(model="gpt-4", endpoint="https://test.openai.azure.com")


def test_azure_endpoint_configuration():
    """
    Test that Azure endpoint configuration works with multiple environment variable names
    """
    # Test with AZURE_ENDPOINT
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test1.openai.azure.com"
    }):
        llm = LLM(model="azure/gpt-4")

        from crewai.llms.providers.azure.completion import AzureCompletion
        assert isinstance(llm, AzureCompletion)
        assert llm.endpoint == "https://test1.openai.azure.com/openai/deployments/gpt-4"

    # Test with AZURE_OPENAI_ENDPOINT
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://test2.openai.azure.com"
    }, clear=True):
        llm = LLM(model="azure/gpt-4")

        assert isinstance(llm, AzureCompletion)
        # Endpoint should be auto-constructed for Azure OpenAI
        assert llm.endpoint == "https://test2.openai.azure.com/openai/deployments/gpt-4"


def test_azure_api_key_configuration():
    """
    Test that API key configuration works from AZURE_API_KEY environment variable
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-azure-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com"
    }):
        llm = LLM(model="azure/gpt-4")

        from crewai.llms.providers.azure.completion import AzureCompletion
        assert isinstance(llm, AzureCompletion)
        assert llm.api_key == "test-azure-key"


def test_azure_model_capabilities():
    """
    Test that model capabilities are correctly identified
    """
    # Test GPT-4 model (supports function calling)
    llm_gpt4 = LLM(model="azure/gpt-4")
    from crewai.llms.providers.azure.completion import AzureCompletion
    assert isinstance(llm_gpt4, AzureCompletion)
    assert llm_gpt4.is_openai_model == True
    assert llm_gpt4.supports_function_calling() == True

    # Test GPT-3.5 model
    llm_gpt35 = LLM(model="azure/gpt-35-turbo")
    assert isinstance(llm_gpt35, AzureCompletion)
    assert llm_gpt35.is_openai_model == True
    assert llm_gpt35.supports_function_calling() == True


def test_azure_completion_params_preparation():
    """
    Test that completion parameters are properly prepared
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        llm = LLM(
            model="azure/gpt-4",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            max_tokens=1000
        )

        from crewai.llms.providers.azure.completion import AzureCompletion
        assert isinstance(llm, AzureCompletion)

        messages = [{"role": "user", "content": "Hello"}]
        params = llm._prepare_completion_params(messages)

        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.3
        assert params["max_tokens"] == 1000


def test_azure_model_detection():
    """
    Test that various Azure model formats are properly detected
    """
    # Test Azure model naming patterns
    azure_test_cases = [
        "azure/gpt-4",
        "azure_openai/gpt-4",
        "azure/gpt-4o",
        "azure/gpt-35-turbo"
    ]

    for model_name in azure_test_cases:
        llm = LLM(model=model_name)
        from crewai.llms.providers.azure.completion import AzureCompletion
        assert isinstance(llm, AzureCompletion), f"Failed for model: {model_name}"


def test_azure_supports_stop_words():
    """
    Test that Azure models support stop sequences
    """
    llm = LLM(model="azure/gpt-4")
    assert llm.supports_stop_words() == True


def test_azure_gpt5_models_do_not_support_stop_words():
    """
    Test that GPT-5 family models do not support stop words.
    GPT-5 models use the Responses API which doesn't support stop sequences.
    See: https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/models-sold-directly-by-azure
    """
    # GPT-5 base models
    gpt5_models = [
        "azure/gpt-5",
        "azure/gpt-5-mini",
        "azure/gpt-5-nano",
        "azure/gpt-5-chat",
        # GPT-5.1 series
        "azure/gpt-5.1",
        "azure/gpt-5.1-chat",
        "azure/gpt-5.1-codex",
        "azure/gpt-5.1-codex-mini",
        # GPT-5.2 series
        "azure/gpt-5.2",
        "azure/gpt-5.2-chat",
    ]

    for model_name in gpt5_models:
        llm = LLM(model=model_name)
        assert llm.supports_stop_words() == False, f"Expected {model_name} to NOT support stop words"


def test_azure_o_series_models_do_not_support_stop_words():
    """
    Test that o-series reasoning models do not support stop words.
    """
    o_series_models = [
        "azure/o1",
        "azure/o1-mini",
        "azure/o3",
        "azure/o3-mini",
        "azure/o4",
        "azure/o4-mini",
    ]

    for model_name in o_series_models:
        llm = LLM(model=model_name)
        assert llm.supports_stop_words() == False, f"Expected {model_name} to NOT support stop words"


def test_azure_responses_api_models_do_not_support_stop_words():
    """
    Test that models using the Responses API do not support stop words.
    """
    responses_api_models = [
        "azure/computer-use-preview",
    ]

    for model_name in responses_api_models:
        llm = LLM(model=model_name)
        assert llm.supports_stop_words() == False, f"Expected {model_name} to NOT support stop words"


def test_azure_stop_words_not_included_for_unsupported_models():
    """
    Test that stop words are not included in completion params for models that don't support them.
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        # Test GPT-5 model - stop should NOT be included even if set
        llm_gpt5 = LLM(
            model="azure/gpt-5-nano",
            stop=["STOP", "END"]
        )
        params = llm_gpt5._prepare_completion_params(
            messages=[{"role": "user", "content": "test"}]
        )
        assert "stop" not in params, "stop should not be included for GPT-5 models"

        # Test regular model - stop SHOULD be included
        llm_gpt4 = LLM(
            model="azure/gpt-4",
            stop=["STOP", "END"]
        )
        params = llm_gpt4._prepare_completion_params(
            messages=[{"role": "user", "content": "test"}]
        )
        assert "stop" in params, "stop should be included for GPT-4 models"
        assert params["stop"] == ["STOP", "END"]


def test_azure_context_window_size():
    """
    Test that Azure models return correct context window sizes
    """
    # Test GPT-4
    llm_gpt4 = LLM(model="azure/gpt-4")
    context_size_gpt4 = llm_gpt4.get_context_window_size()
    assert context_size_gpt4 > 0  # Should return valid context size

    # Test GPT-4o
    llm_gpt4o = LLM(model="azure/gpt-4o")
    context_size_gpt4o = llm_gpt4o.get_context_window_size()
    assert context_size_gpt4o > context_size_gpt4  # GPT-4o has larger context


def test_azure_message_formatting():
    """
    Test that messages are properly formatted for Azure API
    """
    llm = LLM(model="azure/gpt-4")

    # Test message formatting
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    formatted_messages = llm._format_messages_for_azure(test_messages)

    # All messages should be formatted as dictionaries with content
    assert len(formatted_messages) == 4

    # Verify each message is a dict with content
    for msg in formatted_messages:
        assert isinstance(msg, dict)
        assert "content" in msg


def test_azure_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    # Test non-streaming
    llm_no_stream = LLM(model="azure/gpt-4", stream=False)
    assert llm_no_stream.stream == False

    # Test streaming
    llm_stream = LLM(model="azure/gpt-4", stream=True)
    assert llm_stream.stream == True


def test_azure_tool_conversion():
    """
    Test that tools are properly converted to Azure OpenAI format
    """
    llm = LLM(model="azure/gpt-4")

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
    azure_tools = llm._convert_tools_for_interference(crewai_tools)

    assert len(azure_tools) == 1
    # Azure tools should maintain the function calling format
    assert azure_tools[0]["type"] == "function"
    assert azure_tools[0]["function"]["name"] == "test_tool"
    assert azure_tools[0]["function"]["description"] == "A test tool"
    assert "parameters" in azure_tools[0]["function"]


def test_azure_environment_variable_endpoint():
    """
    Test that Azure endpoint is properly loaded from environment
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com"
    }):
        llm = LLM(model="azure/gpt-4")

        assert llm.client is not None
        assert llm.endpoint == "https://test.openai.azure.com/openai/deployments/gpt-4"


def test_azure_token_usage_tracking():
    """
    Test that token usage is properly tracked for Azure responses
    """
    llm = LLM(model="azure/gpt-4")

    # Mock the Azure response with usage information
    with patch.object(llm.client, 'complete') as mock_complete:
        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75
        )
        mock_complete.return_value = mock_response

        result = llm.call("Hello")

        # Verify the response
        assert result == "test response"

        # Verify token usage was extracted
        usage = llm._extract_azure_token_usage(mock_response)
        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 25
        assert usage["total_tokens"] == 75


def test_azure_http_error_handling():
    """
    Test that Azure HTTP errors are properly handled
    """
    from azure.core.exceptions import HttpResponseError

    llm = LLM(model="azure/gpt-4")

    # Mock an HTTP error
    with patch.object(llm.client, 'complete') as mock_complete:
        mock_complete.side_effect = HttpResponseError(message="Rate limit exceeded", response=MagicMock(status_code=429))

        with pytest.raises(HttpResponseError):
            llm.call("Hello")


@pytest.mark.vcr()
def test_azure_streaming_completion():
    """
    Test that streaming completions work properly
    """
    llm = LLM(model="azure/gpt-4o-mini", stream=True)
    result = llm.call("Say hello")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_azure_api_version_default():
    """
    Test that Azure API version defaults correctly
    """
    llm = LLM(model="azure/gpt-4")

    from crewai.llms.providers.azure.completion import AzureCompletion
    assert isinstance(llm, AzureCompletion)
    # Should use default or environment variable
    assert llm.api_version is not None


def test_azure_function_calling_support():
    """
    Test that function calling is supported for OpenAI models
    """
    # Test with GPT-4 (supports function calling)
    llm_gpt4 = LLM(model="azure/gpt-4")
    assert llm_gpt4.supports_function_calling() == True

    # Test with GPT-3.5 (supports function calling)
    llm_gpt35 = LLM(model="azure/gpt-35-turbo")
    assert llm_gpt35.supports_function_calling() == True


def test_azure_openai_endpoint_url_construction():
    """
    Test that Azure OpenAI endpoint URLs are automatically constructed correctly
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test-resource.openai.azure.com"
    }):
        llm = LLM(model="azure/gpt-4o-mini")

        assert "/openai/deployments/gpt-4o-mini" in llm.endpoint
        assert llm.endpoint == "https://test-resource.openai.azure.com/openai/deployments/gpt-4o-mini"
        assert llm.is_azure_openai_endpoint == True


def test_azure_openai_endpoint_url_with_trailing_slash():
    """
    Test that trailing slashes are handled correctly in endpoint URLs
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test-resource.openai.azure.com/"  # trailing slash
    }):
        llm = LLM(model="azure/gpt-4o")

        assert llm.endpoint == "https://test-resource.openai.azure.com/openai/deployments/gpt-4o"
        assert not llm.endpoint.endswith("//")


def test_azure_openai_endpoint_already_complete():
    """
    Test that already complete Azure OpenAI endpoint URLs are not modified
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test-resource.openai.azure.com/openai/deployments/my-deployment"
    }):
        llm = LLM(model="azure/gpt-4")

        assert llm.endpoint == "https://test-resource.openai.azure.com/openai/deployments/my-deployment"
        assert llm.is_azure_openai_endpoint == True


def test_non_azure_openai_endpoint_unchanged():
    """
    Test that non-Azure OpenAI endpoints are not modified
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        llm = LLM(model="azure/mistral-large")

        assert llm.endpoint == "https://models.inference.ai.azure.com"
        assert llm.is_azure_openai_endpoint == False


def test_azure_openai_model_parameter_excluded():
    """
    Test that model parameter is NOT included for Azure OpenAI endpoints
    """

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com/openai/deployments/gpt-4"
    }):
        llm = LLM(model="azure/gpt-4")

        # Prepare params to check model parameter handling
        params = llm._prepare_completion_params(
            messages=[{"role": "user", "content": "test"}]
        )

        # Model parameter should NOT be included for Azure OpenAI endpoints
        assert "model" not in params
        assert "messages" in params
        assert params["stream"] == False


def test_non_azure_openai_model_parameter_included():
    """
    Test that model parameter IS included for non-Azure OpenAI endpoints
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        llm = LLM(model="azure/mistral-large")

        params = llm._prepare_completion_params(
            messages=[{"role": "user", "content": "test"}]
        )

        assert "model" in params
        assert params["model"] == "mistral-large"


def test_azure_message_formatting_with_role():
    """
    Test that messages are formatted with both 'role' and 'content' fields
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    llm = LLM(model="azure/gpt-4")

    # Test with string message
    formatted = llm._format_messages_for_azure("Hello world")
    assert isinstance(formatted, list)
    assert len(formatted) > 0
    assert "role" in formatted[0]
    assert "content" in formatted[0]

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    formatted = llm._format_messages_for_azure(messages)

    for msg in formatted:
        assert "role" in msg
        assert "content" in msg
        assert msg["role"] in ["system", "user", "assistant"]


def test_azure_message_formatting_default_role():
    """
    Test that messages without a role default to 'user'
    """

    llm = LLM(model="azure/gpt-4")

    # Test with message that has role but tests default behavior
    messages = [{"role": "user", "content": "test message"}]
    formatted = llm._format_messages_for_azure(messages)

    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "test message"


def test_azure_endpoint_detection_flags():
    """
    Test that is_azure_openai_endpoint flag is set correctly
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com/openai/deployments/gpt-4"
    }):
        llm_openai = LLM(model="azure/gpt-4")
        assert llm_openai.is_azure_openai_endpoint == True

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        llm_other = LLM(model="azure/mistral-large")
        assert llm_other.is_azure_openai_endpoint == False


def test_azure_improved_error_messages():
    """
    Test that improved error messages are provided for common HTTP errors
    """
    from crewai.llms.providers.azure.completion import AzureCompletion
    from azure.core.exceptions import HttpResponseError

    llm = LLM(model="azure/gpt-4")

    with patch.object(llm.client, 'complete') as mock_complete:
        error_401 = HttpResponseError(message="Unauthorized")
        error_401.status_code = 401
        mock_complete.side_effect = error_401

        with pytest.raises(HttpResponseError):
            llm.call("test")

        error_404 = HttpResponseError(message="Not Found")
        error_404.status_code = 404
        mock_complete.side_effect = error_404

        with pytest.raises(HttpResponseError):
            llm.call("test")

        error_429 = HttpResponseError(message="Rate Limited")
        error_429.status_code = 429
        mock_complete.side_effect = error_429

        with pytest.raises(HttpResponseError):
            llm.call("test")


def test_azure_api_version_properly_passed():
    """
    Test that api_version is properly passed to the client
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_API_VERSION": ""  # Clear env var to test default
    }, clear=False):
        llm = LLM(model="azure/gpt-4", api_version="2024-08-01")
        assert llm.api_version == "2024-08-01"

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com"
    }, clear=True):
        llm_default = LLM(model="azure/gpt-4")
        assert llm_default.api_version == "2024-06-01"  # Current default


def test_azure_timeout_and_max_retries_stored():
    """
    Test that timeout and max_retries parameters are stored
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com"
    }):
        llm = LLM(
            model="azure/gpt-4",
            timeout=60.0,
            max_retries=5
        )

        assert llm.timeout == 60.0
        assert llm.max_retries == 5


def test_azure_complete_params_include_optional_params():
    """
    Test that optional parameters are included in completion params when set
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        llm = LLM(
            model="azure/gpt-4",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            max_tokens=1000,
            stop=["STOP", "END"]
        )

        params = llm._prepare_completion_params(
            messages=[{"role": "user", "content": "test"}]
        )

        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.3
        assert params["max_tokens"] == 1000
        assert params["stop"] == ["STOP", "END"]


def test_azure_endpoint_validation_with_azure_prefix():
    """
    Test that 'azure/' prefix is properly stripped when constructing endpoint
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://test.openai.azure.com"
    }):
        llm = LLM(model="azure/gpt-4o-mini")

        # Should strip 'azure/' prefix and use 'gpt-4o-mini' as deployment name
        assert "gpt-4o-mini" in llm.endpoint
        assert "azure/gpt-4o-mini" not in llm.endpoint


def test_azure_message_formatting_preserves_all_roles():
    """
    Test that all message roles (system, user, assistant) are preserved correctly
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    llm = LLM(model="azure/gpt-4")

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "Another user message"}
    ]

    formatted = llm._format_messages_for_azure(messages)

    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == "System message"
    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"] == "User message"
    assert formatted[2]["role"] == "assistant"
    assert formatted[2]["content"] == "Assistant message"
    assert formatted[3]["role"] == "user"
    assert formatted[3]["content"] == "Another user message"


def test_azure_deepseek_model_support():
    """
    Test that DeepSeek and other non-OpenAI models work correctly with Azure AI Inference
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        # Test DeepSeek model
        llm_deepseek = LLM(model="azure/deepseek-chat")

        # Endpoint should not be modified for non-OpenAI endpoints
        assert llm_deepseek.endpoint == "https://models.inference.ai.azure.com"
        assert llm_deepseek.is_azure_openai_endpoint == False

        # Model parameter should be included in completion params
        params = llm_deepseek._prepare_completion_params(
            messages=[{"role": "user", "content": "test"}]
        )
        assert "model" in params
        assert params["model"] == "deepseek-chat"

        # Should not be detected as OpenAI model (no function calling)
        assert llm_deepseek.is_openai_model == False
        assert llm_deepseek.supports_function_calling() == False


def test_azure_mistral_and_other_models():
    """
    Test that various non-OpenAI models (Mistral, Llama, etc.) work with Azure AI Inference
    """
    test_models = [
        "mistral-large-latest",
        "llama-3-70b-instruct",
        "cohere-command-r-plus"
    ]

    for model_name in test_models:
        with patch.dict(os.environ, {
            "AZURE_API_KEY": "test-key",
            "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
        }):
            llm = LLM(model=f"azure/{model_name}")

            # Verify endpoint is not modified
            assert llm.endpoint == "https://models.inference.ai.azure.com"
            assert llm.is_azure_openai_endpoint == False

            # Verify model parameter is included
            params = llm._prepare_completion_params(
                messages=[{"role": "user", "content": "test"}]
            )
            assert "model" in params
            assert params["model"] == model_name


def test_azure_completion_params_preparation_with_drop_params():
    """
    Test that completion parameters are properly prepared with drop paramaeters attribute respected
    """
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "test-key",
        "AZURE_ENDPOINT": "https://models.inference.ai.azure.com"
    }):
        llm = LLM(
            model="azure/o4-mini",
            drop_params=True,
            additional_drop_params=["stop"],
            max_tokens=1000
        )

        from crewai.llms.providers.azure.completion import AzureCompletion
        assert isinstance(llm, AzureCompletion)

        messages = [{"role": "user", "content": "Hello"}]
        params = llm._prepare_completion_params(messages)

        assert params.get('stop') == None


@pytest.mark.vcr()
def test_azure_streaming_returns_usage_metrics():
    """
    Test that Azure streaming calls return proper token usage metrics.
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the capital of Spain",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="azure/gpt-4o-mini", stream=True),
        verbose=True,
    )

    task = Task(
        description="What is the capital of Spain?",
        expected_output="The capital of Spain",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    assert result.token_usage is not None
    assert result.token_usage.total_tokens > 0
    assert result.token_usage.prompt_tokens > 0
    assert result.token_usage.completion_tokens > 0
    assert result.token_usage.successful_requests >= 1


# =============================================================================
# Agent Kickoff Structured Output Tests
# =============================================================================


@pytest.mark.vcr()
def test_azure_agent_kickoff_structured_output_without_tools():
    """
    Test that agent kickoff returns structured output without tools.
    This tests native structured output handling for Azure OpenAI models.
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
        llm=LLM(model="azure/gpt-4o-mini"),
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
def test_azure_agent_kickoff_structured_output_with_tools():
    """
    Test that agent kickoff returns structured output after using tools.
    This tests post-tool-call structured output handling for Azure OpenAI models.
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
        llm=LLM(model="azure/gpt-4o-mini"),
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



def test_azure_stop_words_not_applied_to_structured_output():
    """
    Test that stop words are NOT applied when response_model is provided.
    This ensures JSON responses containing stop word patterns (like "Observation:")
    are not truncated, which would cause JSON validation to fail.
    """
    from pydantic import BaseModel, Field
    from crewai.llms.providers.azure.completion import AzureCompletion

    class ResearchResult(BaseModel):
        """Research result that may contain stop word patterns in string fields."""

        finding: str = Field(description="The research finding")
        observation: str = Field(description="Observation about the finding")

    # Create AzureCompletion instance with stop words configured
    llm = AzureCompletion(
        model="gpt-4",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
        stop=["Observation:", "Final Answer:"],  # Common stop words
    )

    # JSON response that contains a stop word pattern in a string field
    # Without the fix, this would be truncated at "Observation:" breaking the JSON
    json_response = '{"finding": "The data shows growth", "observation": "Observation: This confirms the hypothesis"}'

    with patch.object(llm.client, 'complete') as mock_complete:
        mock_message = MagicMock()
        mock_message.content = json_response
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        mock_complete.return_value = mock_response

        # Call with response_model - stop words should NOT be applied
        result = llm.call(
            messages=[{"role": "user", "content": "Analyze the data"}],
            response_model=ResearchResult,
        )

        # Should successfully parse the full JSON without truncation
        assert isinstance(result, ResearchResult)
        assert result.finding == "The data shows growth"
        # The observation field should contain the full text including "Observation:"
        assert "Observation:" in result.observation


def test_azure_stop_words_still_applied_to_regular_responses():
    """
    Test that stop words ARE still applied for regular (non-structured) responses.
    This ensures the fix didn't break normal stop word behavior.
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    # Create AzureCompletion instance with stop words configured
    llm = AzureCompletion(
        model="gpt-4",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
        stop=["Observation:", "Final Answer:"],
    )

    # Response that contains a stop word - should be truncated
    response_with_stop_word = "I need to search for more information.\n\nAction: search\nObservation: Found results"

    with patch.object(llm.client, 'complete') as mock_complete:
        mock_message = MagicMock()
        mock_message.content = response_with_stop_word
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        mock_complete.return_value = mock_response

        # Call WITHOUT response_model - stop words SHOULD be applied
        result = llm.call(
            messages=[{"role": "user", "content": "Search for something"}],
        )

        # Response should be truncated at the stop word
        assert "Observation:" not in result
        assert "Found results" not in result
        assert "I need to search for more information" in result


# =============================================================================
# Azure Responses API Tests
# =============================================================================


def test_azure_responses_api_initialization():
    """Test that Azure Responses API can be initialized with api='responses'."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            instructions="You are a helpful assistant.",
            store=True,
        )

        assert completion.api == "responses"
        assert completion.instructions == "You are a helpful assistant."
        assert completion.store is True
        assert completion.model == "gpt-4o"


def test_azure_responses_api_default_is_completions():
    """Test that the default API is 'completions' for backward compatibility."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    completion = AzureCompletion(
        model="gpt-4o",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
    )

    assert completion.api == "completions"


def test_azure_responses_api_prepare_params():
    """Test that Responses API params are prepared correctly."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            instructions="Base instructions.",
            store=True,
            temperature=0.7,
        )

        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "Hello!"},
        ]

        params = completion._prepare_responses_params(messages)

        assert params["model"] == "gpt-4o"
        assert "Base instructions." in params["instructions"]
        assert "System message." in params["instructions"]
        assert params["store"] is True
        assert params["temperature"] == 0.7
        assert params["input"] == [{"role": "user", "content": "Hello!"}]


def test_azure_responses_api_tool_format():
    """Test that tools are converted to Responses API format (internally-tagged)."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

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

        responses_tools = completion._convert_tools_for_responses(tools)

        assert len(responses_tools) == 1
        tool = responses_tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get the weather for a location"
        assert "parameters" in tool
        assert "function" not in tool


def test_azure_responses_api_structured_output_format():
    """Test that structured outputs use text.format for Responses API."""
    from pydantic import BaseModel
    from crewai.llms.providers.azure.completion import AzureCompletion

    class Person(BaseModel):
        name: str
        age: int

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        messages = [{"role": "user", "content": "Extract: Jane, 25"}]
        params = completion._prepare_responses_params(messages, response_model=Person)

        assert "text" in params
        assert "format" in params["text"]
        assert params["text"]["format"]["type"] == "json_schema"
        assert params["text"]["format"]["name"] == "Person"
        assert params["text"]["format"]["strict"] is True


def test_azure_responses_api_with_previous_response_id():
    """Test that previous_response_id is passed for multi-turn conversations."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            previous_response_id="resp_abc123",
            store=True,
        )

        messages = [{"role": "user", "content": "Continue our conversation."}]
        params = completion._prepare_responses_params(messages)

        assert params["previous_response_id"] == "resp_abc123"
        assert params["store"] is True


def test_azure_responses_api_call_routing():
    """Test that call() routes to the correct API based on the api parameter."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    completion_completions = AzureCompletion(
        model="gpt-4o",
        api="completions",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
    )

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion_responses = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

    with patch.object(
        completion_completions, "_handle_completion", return_value="completions result"
    ) as mock_completions:
        with patch.object(completion_completions, "_format_messages_for_azure", return_value=[{"role": "user", "content": "Hello"}]):
            result = completion_completions.call("Hello")
            mock_completions.assert_called_once()
            assert result == "completions result"

    with patch.object(
        completion_responses, "_call_responses", return_value="responses result"
    ) as mock_responses:
        with patch.object(completion_responses, "_format_messages_for_azure", return_value=[{"role": "user", "content": "Hello"}]):
            result = completion_responses.call("Hello")
            mock_responses.assert_called_once()
            assert result == "responses result"


def test_azure_responses_api_builtin_tools_param():
    """Test that builtin_tools parameter is properly configured."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            builtin_tools=["web_search", "code_interpreter"],
        )

        assert completion.builtin_tools == ["web_search", "code_interpreter"]

        messages = [{"role": "user", "content": "Test"}]
        params = completion._prepare_responses_params(messages)

        assert "tools" in params
        tool_types = [t["type"] for t in params["tools"]]
        assert "web_search_preview" in tool_types
        assert "code_interpreter" in tool_types


def test_azure_responses_api_builtin_tools_with_custom_tools():
    """Test that builtin_tools can be combined with custom function tools."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
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
        params = completion._prepare_responses_params(messages, tools=custom_tools)

        assert len(params["tools"]) == 2
        tool_types = [t.get("type") for t in params["tools"]]
        assert "web_search_preview" in tool_types
        assert "function" in tool_types


def test_azure_responses_api_parse_tool_outputs_param():
    """Test that parse_tool_outputs parameter is properly configured."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            parse_tool_outputs=True,
        )

        assert completion.parse_tool_outputs is True


def test_azure_responses_api_parse_tool_outputs_default_false():
    """Test that parse_tool_outputs defaults to False."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        assert completion.parse_tool_outputs is False


# =============================================================================
# Auto-Chaining Tests (Azure Responses API)
# =============================================================================


def test_azure_responses_api_auto_chain_param():
    """Test that auto_chain parameter is properly configured."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain=True,
        )

        assert completion.auto_chain is True
        assert completion._last_response_id is None


def test_azure_responses_api_auto_chain_default_false():
    """Test that auto_chain defaults to False."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        assert completion.auto_chain is False


def test_azure_responses_api_last_response_id_property():
    """Test last_response_id property."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain=True,
        )

        # Initially None
        assert completion.last_response_id is None

        # Simulate setting the internal value
        completion._last_response_id = "resp_test_123"
        assert completion.last_response_id == "resp_test_123"


def test_azure_responses_api_reset_chain():
    """Test reset_chain() method clears the response ID."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain=True,
        )

        # Set a response ID
        completion._last_response_id = "resp_test_123"
        assert completion.last_response_id == "resp_test_123"

        # Reset the chain
        completion.reset_chain()
        assert completion.last_response_id is None


def test_azure_responses_api_auto_chain_prepare_params():
    """Test that _prepare_responses_params uses auto-chained response ID."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain=True,
        )

        # No previous response ID yet
        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert "previous_response_id" not in params

        # Set a previous response ID
        completion._last_response_id = "resp_previous_123"
        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert params.get("previous_response_id") == "resp_previous_123"


def test_azure_responses_api_explicit_previous_response_id_takes_precedence():
    """Test that explicit previous_response_id overrides auto-chained ID."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain=True,
            previous_response_id="resp_explicit_456",
        )

        # Set an auto-chained response ID
        completion._last_response_id = "resp_auto_123"

        # Explicit should take precedence
        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert params.get("previous_response_id") == "resp_explicit_456"


def test_azure_responses_api_auto_chain_disabled_no_tracking():
    """Test that response ID is not tracked when auto_chain is False."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain=False,
        )

        # Even with a "previous" response ID set internally, params shouldn't use it
        completion._last_response_id = "resp_should_not_use"
        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert "previous_response_id" not in params


# =============================================================================
# Encrypted Reasoning for ZDR (Zero Data Retention) Tests
# =============================================================================


def test_azure_responses_api_auto_chain_reasoning_param():
    """Test that auto_chain_reasoning parameter is properly configured."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=True,
        )

        assert completion.auto_chain_reasoning is True
        assert completion._last_reasoning_items == []


def test_azure_responses_api_auto_chain_reasoning_default_false():
    """Test that auto_chain_reasoning defaults to False."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        assert completion.auto_chain_reasoning is False


def test_azure_responses_api_last_reasoning_items_property():
    """Test last_reasoning_items property."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=True,
        )

        # Initially empty
        assert completion.last_reasoning_items == []

        # Simulate setting the internal value
        mock_items = [{"id": "rs_test_123", "type": "reasoning"}]
        completion._last_reasoning_items = mock_items
        assert completion.last_reasoning_items == mock_items


def test_azure_responses_api_reset_reasoning_chain():
    """Test reset_reasoning_chain() method clears reasoning items."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=True,
        )

        # Set reasoning items
        mock_items = [{"id": "rs_test_123", "type": "reasoning"}]
        completion._last_reasoning_items = mock_items
        assert completion.last_reasoning_items == mock_items

        # Reset the reasoning chain
        completion.reset_reasoning_chain()
        assert completion.last_reasoning_items == []


def test_azure_responses_api_auto_chain_reasoning_adds_include():
    """Test that auto_chain_reasoning adds reasoning.encrypted_content to include."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=True,
        )

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert "include" in params
        assert "reasoning.encrypted_content" in params["include"]


def test_azure_responses_api_auto_chain_reasoning_preserves_existing_include():
    """Test that auto_chain_reasoning preserves existing include items."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=True,
            include=["file_search_call.results"],
        )

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert "include" in params
        assert "reasoning.encrypted_content" in params["include"]
        assert "file_search_call.results" in params["include"]


def test_azure_responses_api_auto_chain_reasoning_no_duplicate_include():
    """Test that reasoning.encrypted_content is not duplicated if already in include."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=True,
            include=["reasoning.encrypted_content"],
        )

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert "include" in params
        # Should only appear once
        assert params["include"].count("reasoning.encrypted_content") == 1


def test_azure_responses_api_auto_chain_reasoning_prepends_to_input():
    """Test that stored reasoning items are prepended to input."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=True,
        )

        # Simulate stored reasoning items
        mock_reasoning = MagicMock()
        mock_reasoning.type = "reasoning"
        mock_reasoning.id = "rs_test_123"
        completion._last_reasoning_items = [mock_reasoning]

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])

        # Input should have reasoning item first, then the message
        assert len(params["input"]) == 2
        assert params["input"][0] == mock_reasoning
        assert params["input"][1]["role"] == "user"


def test_azure_responses_api_auto_chain_reasoning_disabled_no_include():
    """Test that reasoning.encrypted_content is not added when auto_chain_reasoning is False."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=False,
        )

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        # Should not have include at all (unless explicitly set)
        assert "include" not in params or "reasoning.encrypted_content" not in params.get("include", [])


def test_azure_responses_api_auto_chain_reasoning_disabled_no_prepend():
    """Test that reasoning items are not prepended when auto_chain_reasoning is False."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain_reasoning=False,
        )

        # Even with stored reasoning items, they should not be prepended
        mock_reasoning = MagicMock()
        mock_reasoning.type = "reasoning"
        completion._last_reasoning_items = [mock_reasoning]

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])

        # Input should only have the message, not the reasoning item
        assert len(params["input"]) == 1
        assert params["input"][0]["role"] == "user"


def test_azure_responses_api_both_auto_chains_work_together():
    """Test that auto_chain and auto_chain_reasoning can be used together."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            auto_chain=True,
            auto_chain_reasoning=True,
        )

        assert completion.auto_chain is True
        assert completion.auto_chain_reasoning is True
        assert completion._last_response_id is None
        assert completion._last_reasoning_items == []

        # Set both internal values
        completion._last_response_id = "resp_123"
        mock_reasoning = MagicMock()
        mock_reasoning.type = "reasoning"
        completion._last_reasoning_items = [mock_reasoning]

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])

        # Both should be applied
        assert params.get("previous_response_id") == "resp_123"
        assert "reasoning.encrypted_content" in params["include"]
        assert len(params["input"]) == 2  # Reasoning item + message


def test_azure_responses_api_max_completion_tokens():
    """Test that max_completion_tokens is mapped to max_output_tokens."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            max_completion_tokens=4096,
        )

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert params["max_output_tokens"] == 4096


def test_azure_responses_api_seed_param():
    """Test that seed parameter is passed through."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            seed=42,
        )

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert params["seed"] == 42


def test_azure_responses_api_reasoning_effort_param():
    """Test that reasoning_effort parameter is passed through."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            reasoning_effort="high",
        )

        params = completion._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
        assert params["reasoning"] == {"effort": "high"}


def test_azure_responses_api_init_responses_clients():
    """Test that _init_responses_clients creates OpenAI AzureOpenAI clients."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients") as mock_init:
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        # _init_responses_clients should be called during __init__
        mock_init.assert_called_once()


def test_azure_responses_api_system_message_extraction():
    """Test that system messages are extracted to instructions for Responses API."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        params = completion._prepare_responses_params(messages)

        # System message should be extracted to instructions
        assert params["instructions"] == "You are a helpful assistant."
        # Non-system messages should be in input
        assert len(params["input"]) == 3
        assert params["input"][0]["role"] == "user"
        assert params["input"][1]["role"] == "assistant"
        assert params["input"][2]["role"] == "user"


def test_azure_responses_api_multiple_system_messages_merged():
    """Test that multiple system messages are merged into instructions."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    with patch("crewai.llms.providers.azure.completion.AzureCompletion._init_responses_clients"):
        completion = AzureCompletion(
            model="gpt-4o",
            api="responses",
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            instructions="Base instructions.",
        )

        messages = [
            {"role": "system", "content": "System context."},
            {"role": "user", "content": "Hello!"},
        ]

        params = completion._prepare_responses_params(messages)

        # Both base instructions and system message should be merged
        assert "Base instructions." in params["instructions"]
        assert "System context." in params["instructions"]
