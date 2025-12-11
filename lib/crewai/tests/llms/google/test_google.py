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
def mock_google_api_key():
    """Automatically mock GOOGLE_API_KEY for all tests in this module."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        yield


def test_gemini_completion_is_used_when_google_provider():
    """
    Test that GeminiCompletion from completion.py is used when LLM uses provider 'google'
    """
    llm = LLM(model="google/gemini-2.0-flash-001")

    assert llm.__class__.__name__ == "GeminiCompletion"
    assert llm.provider == "gemini"
    assert llm.model == "gemini-2.0-flash-001"


def test_gemini_completion_is_used_when_gemini_provider():
    """
    Test that GeminiCompletion is used when provider is 'gemini'
    """
    llm = LLM(model="gemini/gemini-2.0-flash-001")

    from crewai.llms.providers.gemini.completion import GeminiCompletion
    assert isinstance(llm, GeminiCompletion)
    assert llm.provider == "gemini"
    assert llm.model == "gemini-2.0-flash-001"




def test_gemini_tool_use_conversation_flow():
    """
    Test that the Gemini completion properly handles tool use conversation flow
    """
    from unittest.mock import Mock, patch
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    # Create GeminiCompletion instance
    completion = GeminiCompletion(model="gemini-2.0-flash-001")

    # Mock tool function
    def mock_weather_tool(location: str) -> str:
        return f"The weather in {location} is sunny and 75°F"

    available_functions = {"get_weather": mock_weather_tool}

    # Mock the Google Gemini client responses
    with patch.object(completion.client.models, 'generate_content') as mock_generate:
        # Mock function call in response
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "San Francisco"}

        mock_part = Mock()
        mock_part.function_call = mock_function_call

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Based on the weather data, it's a beautiful day in San Francisco with sunny skies and 75°F temperature."
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_generate.return_value = mock_response

        # Test the call
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        result = completion.call(
            messages=messages,
            available_functions=available_functions
        )

        # Verify the tool was executed and returned the result
        assert result == "The weather in San Francisco is sunny and 75°F"

        # Verify that the API was called
        assert mock_generate.called


def test_gemini_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using Google provider
    """
    module_name = "crewai.llms.providers.gemini.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="google/gemini-2.0-flash-001")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'GeminiCompletion')


def test_native_gemini_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native Gemini completion fails.

    With the new behavior, when a native provider is in SUPPORTED_NATIVE_PROVIDERS
    but fails to instantiate, we raise an ImportError instead of silently falling back.
    This provides clearer error messages to users about missing dependencies.
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native Google Gen AI SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should raise ImportError with clear message
        with pytest.raises(ImportError) as excinfo:
            LLM(model="google/gemini-2.0-flash-001")

        # Verify the error message is helpful
        assert "Error importing native provider" in str(excinfo.value)
        assert "Native Google Gen AI SDK failed" in str(excinfo.value)


def test_gemini_completion_initialization_parameters():
    """
    Test that GeminiCompletion is initialized with correct parameters
    """
    llm = LLM(
        model="google/gemini-2.0-flash-001",
        temperature=0.7,
        max_output_tokens=2000,
        top_p=0.9,
        top_k=40,
        api_key="test-key"
    )

    from crewai.llms.providers.gemini.completion import GeminiCompletion
    assert isinstance(llm, GeminiCompletion)
    assert llm.model == "gemini-2.0-flash-001"
    assert llm.temperature == 0.7
    assert llm.max_output_tokens == 2000
    assert llm.top_p == 0.9
    assert llm.top_k == 40


def test_gemini_specific_parameters():
    """
    Test Gemini-specific parameters like stop_sequences, streaming, and safety settings
    """
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
    }

    llm = LLM(
        model="google/gemini-2.0-flash-001",
        stop_sequences=["Human:", "Assistant:"],
        stream=True,
        safety_settings=safety_settings,
        project="test-project",
        location="us-central1"
    )

    from crewai.llms.providers.gemini.completion import GeminiCompletion
    assert isinstance(llm, GeminiCompletion)
    assert llm.stop_sequences == ["Human:", "Assistant:"]
    assert llm.stream == True
    assert llm.safety_settings == safety_settings
    assert llm.project == "test-project"
    assert llm.location == "us-central1"


def test_gemini_completion_call():
    """
    Test that GeminiCompletion call method works
    """
    llm = LLM(model="google/gemini-2.0-flash-001")

    # Mock the call method on the instance
    with patch.object(llm, 'call', return_value="Hello! I'm Gemini, ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm Gemini, ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


def test_gemini_completion_called_during_crew_execution():
    """
    Test that GeminiCompletion.call is actually invoked when running a crew
    """
    # Create the LLM instance first
    gemini_llm = LLM(model="google/gemini-2.0-flash-001")

    # Mock the call method on the specific instance
    with patch.object(gemini_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:

        # Create agent with explicit LLM configuration
        agent = Agent(
            role="Research Assistant",
            goal="Find population info",
            backstory="You research populations.",
            llm=gemini_llm,
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


def test_gemini_completion_call_arguments():
    """
    Test that GeminiCompletion.call is invoked with correct arguments
    """
    # Create LLM instance first
    gemini_llm = LLM(model="google/gemini-2.0-flash-001")

    # Mock the instance method
    with patch.object(gemini_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed successfully."

        agent = Agent(
            role="Test Agent",
            goal="Complete a simple task",
            backstory="You are a test agent.",
            llm=gemini_llm  # Use same instance
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


def test_multiple_gemini_calls_in_crew():
    """
    Test that GeminiCompletion.call is invoked multiple times for multiple tasks
    """
    # Create LLM instance first
    gemini_llm = LLM(model="google/gemini-2.0-flash-001")

    # Mock the instance method
    with patch.object(gemini_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent = Agent(
            role="Multi-task Agent",
            goal="Complete multiple tasks",
            backstory="You can handle multiple tasks.",
            llm=gemini_llm  # Use same instance
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


def test_gemini_completion_with_tools():
    """
    Test that GeminiCompletion.call is invoked with tools when agent has tools
    """
    from crewai.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing"""
        return f"Tool result for: {query}"

    # Create LLM instance first
    gemini_llm = LLM(model="google/gemini-2.0-flash-001")

    # Mock the instance method
    with patch.object(gemini_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed with tools."

        agent = Agent(
            role="Tool User",
            goal="Use tools to complete tasks",
            backstory="You can use tools.",
            llm=gemini_llm,  # Use same instance
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


def test_gemini_raises_error_when_model_not_supported():
    """Test that GeminiCompletion raises ValueError when model not supported"""

    # Mock the Google client to raise an error
    with patch('crewai.llms.providers.gemini.completion.genai') as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        from google.genai.errors import ClientError  # type: ignore

        mock_response = MagicMock()
        mock_response.body_segments = [{
            'error': {
                'code': 404,
                'message': 'models/model-doesnt-exist is not found for API version v1beta, or is not supported for generateContent.',
                'status': 'NOT_FOUND'
            }
        }]
        mock_response.status_code = 404

        mock_client.models.generate_content.side_effect = ClientError(404, mock_response)

        llm = LLM(model="google/model-doesnt-exist")

        with pytest.raises(Exception):  # Should raise some error for unsupported model
            llm.call("Hello")


def test_gemini_vertex_ai_setup():
    """
    Test that Vertex AI configuration is properly handled
    """
    with patch.dict(os.environ, {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "us-west1"
    }):
        llm = LLM(
            model="google/gemini-2.0-flash-001",
            project="test-project",
            location="us-west1"
        )

        from crewai.llms.providers.gemini.completion import GeminiCompletion
        assert isinstance(llm, GeminiCompletion)

        assert llm.project == "test-project"
        assert llm.location == "us-west1"


def test_gemini_api_key_configuration():
    """
    Test that API key configuration works for both GOOGLE_API_KEY and GEMINI_API_KEY
    """
    # Test with GOOGLE_API_KEY
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}):
        llm = LLM(model="google/gemini-2.0-flash-001")

        from crewai.llms.providers.gemini.completion import GeminiCompletion
        assert isinstance(llm, GeminiCompletion)
        assert llm.api_key == "test-google-key"

    # Test with GEMINI_API_KEY
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}, clear=True):
        llm = LLM(model="google/gemini-2.0-flash-001")

        assert isinstance(llm, GeminiCompletion)
        assert llm.api_key == "test-gemini-key"


def test_gemini_model_capabilities():
    """
    Test that model capabilities are correctly identified
    """
    # Test Gemini 2.0 model
    llm_2_0 = LLM(model="google/gemini-2.0-flash-001")
    from crewai.llms.providers.gemini.completion import GeminiCompletion
    assert isinstance(llm_2_0, GeminiCompletion)
    assert llm_2_0.supports_tools == True

    # Test Gemini 1.5 model
    llm_1_5 = LLM(model="google/gemini-1.5-pro")
    assert isinstance(llm_1_5, GeminiCompletion)
    assert llm_1_5.supports_tools == True


def test_gemini_generation_config():
    """
    Test that generation config is properly prepared
    """
    llm = LLM(
        model="google/gemini-2.0-flash-001",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=1000
    )

    from crewai.llms.providers.gemini.completion import GeminiCompletion
    assert isinstance(llm, GeminiCompletion)

    # Test config preparation
    config = llm._prepare_generation_config()

    # Verify config has the expected parameters
    assert hasattr(config, 'temperature') or 'temperature' in str(config)
    assert hasattr(config, 'top_p') or 'top_p' in str(config)
    assert hasattr(config, 'top_k') or 'top_k' in str(config)
    assert hasattr(config, 'max_output_tokens') or 'max_output_tokens' in str(config)


def test_gemini_model_detection():
    """
    Test that various Gemini model formats are properly detected
    """
    # Test Gemini model naming patterns that actually work with provider detection
    gemini_test_cases = [
        "google/gemini-2.0-flash-001",
        "gemini/gemini-2.0-flash-001",
        "google/gemini-1.5-pro",
        "gemini/gemini-1.5-flash"
    ]

    for model_name in gemini_test_cases:
        llm = LLM(model=model_name)
        from crewai.llms.providers.gemini.completion import GeminiCompletion
        assert isinstance(llm, GeminiCompletion), f"Failed for model: {model_name}"


def test_gemini_supports_stop_words():
    """
    Test that Gemini models support stop sequences
    """
    llm = LLM(model="google/gemini-2.0-flash-001")
    assert llm.supports_stop_words() == True


def test_gemini_context_window_size():
    """
    Test that Gemini models return correct context window sizes
    """
    # Test Gemini 2.0 Flash
    llm_2_0 = LLM(model="google/gemini-2.0-flash-001")
    context_size_2_0 = llm_2_0.get_context_window_size()
    assert context_size_2_0 > 500000  # Should be substantial (1M tokens)

    # Test Gemini 1.5 Pro
    llm_1_5 = LLM(model="google/gemini-1.5-pro")
    context_size_1_5 = llm_1_5.get_context_window_size()
    assert context_size_1_5 > 1000000  # Should be very large (2M tokens)


def test_gemini_message_formatting():
    """
    Test that messages are properly formatted for Gemini API
    """
    llm = LLM(model="google/gemini-2.0-flash-001")

    # Test message formatting
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    formatted_contents, system_instruction = llm._format_messages_for_gemini(test_messages)

    # System message should be extracted
    assert system_instruction == "You are a helpful assistant."

    # Remaining messages should be Content objects
    assert len(formatted_contents) >= 3  # Should have user, model, user messages

    # First content should be user role
    assert formatted_contents[0].role == "user"
    # Second should be model (converted from assistant)
    assert formatted_contents[1].role == "model"


def test_gemini_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    # Test non-streaming
    llm_no_stream = LLM(model="google/gemini-2.0-flash-001", stream=False)
    assert llm_no_stream.stream == False

    # Test streaming
    llm_stream = LLM(model="google/gemini-2.0-flash-001", stream=True)
    assert llm_stream.stream == True


def test_gemini_tool_conversion():
    """
    Test that tools are properly converted to Gemini format
    """
    llm = LLM(model="google/gemini-2.0-flash-001")

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
    gemini_tools = llm._convert_tools_for_interference(crewai_tools)

    assert len(gemini_tools) == 1
    # Gemini tools are Tool objects with function_declarations
    assert hasattr(gemini_tools[0], 'function_declarations')
    assert len(gemini_tools[0].function_declarations) == 1

    func_decl = gemini_tools[0].function_declarations[0]
    assert func_decl.name == "test_tool"
    assert func_decl.description == "A test tool"


def test_gemini_environment_variable_api_key():
    """
    Test that Google API key is properly loaded from environment
    """
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}):
        llm = LLM(model="google/gemini-2.0-flash-001")

        assert llm.client is not None
        assert hasattr(llm.client, 'models')
        assert llm.api_key == "test-google-key"


def test_gemini_token_usage_tracking():
    """
    Test that token usage is properly tracked for Gemini responses
    """
    llm = LLM(model="google/gemini-2.0-flash-001")

    # Mock the Gemini response with usage information
    with patch.object(llm.client.models, 'generate_content') as mock_generate:
        mock_response = MagicMock()
        mock_response.text = "test response"
        mock_response.candidates = []
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=25,
            total_token_count=75
        )
        mock_generate.return_value = mock_response

        result = llm.call("Hello")

        # Verify the response
        assert result == "test response"

        # Verify token usage was extracted
        usage = llm._extract_token_usage(mock_response)
        assert usage["prompt_token_count"] == 50
        assert usage["candidates_token_count"] == 25
        assert usage["total_token_count"] == 75
        assert usage["total_tokens"] == 75


def test_gemini_stop_sequences_sync():
    """Test that stop and stop_sequences attributes stay synchronized."""
    llm = LLM(model="google/gemini-2.0-flash-001")

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


def test_gemini_stop_sequences_sent_to_api():
    """Test that stop_sequences are properly sent to the Gemini API."""
    llm = LLM(model="google/gemini-2.0-flash-001")

    # Set stop sequences via the stop attribute (simulating CrewAgentExecutor)
    llm.stop = ["\nObservation:", "\nThought:"]

    # Patch the API call to capture parameters without making real call
    with patch.object(llm.client.models, 'generate_content') as mock_generate:
        mock_response = MagicMock()
        mock_response.text = "Hello"
        mock_response.candidates = []
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15
        )
        mock_generate.return_value = mock_response

        llm.call("Say hello in one word")

        # Verify stop_sequences were passed to the API in the config
        call_kwargs = mock_generate.call_args[1]
        assert "config" in call_kwargs
        # The config object should have stop_sequences set
        config = call_kwargs["config"]
        # Check if the config has stop_sequences attribute
        assert hasattr(config, 'stop_sequences') or 'stop_sequences' in config.__dict__
        if hasattr(config, 'stop_sequences'):
            assert config.stop_sequences == ["\nObservation:", "\nThought:"]


@pytest.mark.vcr()
@pytest.mark.skip(reason="VCR cannot replay SSE streaming responses")
def test_google_streaming_returns_usage_metrics():
    """
    Test that Google Gemini streaming calls return proper token usage metrics.
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the capital of Japan",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="gemini/gemini-2.0-flash-exp", stream=True),
        verbose=True,
    )

    task = Task(
        description="What is the capital of Japan?",
        expected_output="The capital of Japan",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    assert result.token_usage is not None
    assert result.token_usage.total_tokens > 0
    assert result.token_usage.prompt_tokens > 0
    assert result.token_usage.completion_tokens > 0
    assert result.token_usage.successful_requests >= 1
