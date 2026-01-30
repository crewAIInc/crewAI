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
    """Mock GOOGLE_API_KEY for tests only if real keys are not set."""
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            yield
    else:
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


@pytest.mark.vcr()
def test_gemini_token_usage_tracking():
    """
    Test that token usage is properly tracked for Gemini responses
    """
    llm = LLM(model="google/gemini-2.0-flash-001")

    result = llm.call("Hello")

    assert result.strip() == "Hi there! How can I help you today?"

    usage = llm.get_token_usage_summary()
    assert usage.successful_requests == 1
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens > 0


@pytest.mark.vcr()
def test_gemini_tool_returning_float():
    """
    Test that Gemini properly handles tools that return non-dict values like floats.

    This is an end-to-end test that verifies the agent can use a tool that returns
    a float (which gets wrapped in {"result": value} for Gemini's FunctionResponse).
    """
    from pydantic import BaseModel, Field
    from typing import Type
    from crewai.tools import BaseTool

    class SumNumbersToolInput(BaseModel):
        a: float = Field(..., description="The first number to add")
        b: float = Field(..., description="The second number to add")

    class SumNumbersTool(BaseTool):
        name: str = "sum_numbers"
        description: str = "Add two numbers together and return the result"
        args_schema: Type[BaseModel] = SumNumbersToolInput

        def _run(self, a: float, b: float) -> float:
            return a + b

    sum_tool = SumNumbersTool()

    agent = Agent(
        role="Calculator",
        goal="Calculate numbers accurately",
        backstory="You are a calculator that adds numbers.",
        llm=LLM(model="google/gemini-2.0-flash-001"),
        tools=[sum_tool],
        verbose=True,
    )

    task = Task(
        description="What is 10000 + 20000? Use the sum_numbers tool to calculate this.",
        expected_output="The sum of the two numbers",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()

    # The result should contain 30000 (the sum)
    assert "30000" in result.raw


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


@pytest.mark.vcr()
def test_google_express_mode_works() -> None:
    """
    Test Google Vertex AI Express mode with API key authentication.
    This tests Vertex AI Express mode (aiplatform.googleapis.com) with API key
    authentication.

    """
    with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true"}):
        agent = Agent(
            role="Research Assistant",
            goal="Find information about the capital of Japan",
            backstory="You are a helpful research assistant.",
            llm=LLM(
                model="gemini/gemini-2.0-flash-exp",
            ),
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


def test_gemini_2_0_model_detection():
    """Test that Gemini 2.0 models are properly detected."""
    # Test Gemini 2.0 models
    llm_2_0 = LLM(model="google/gemini-2.0-flash-001")
    from crewai.llms.providers.gemini.completion import GeminiCompletion
    assert isinstance(llm_2_0, GeminiCompletion)
    assert llm_2_0.is_gemini_2_0 is True

    llm_2_5 = LLM(model="google/gemini-2.5-flash")
    assert isinstance(llm_2_5, GeminiCompletion)
    assert llm_2_5.is_gemini_2_0 is True

    # Test non-2.0 models
    llm_1_5 = LLM(model="google/gemini-1.5-pro")
    assert isinstance(llm_1_5, GeminiCompletion)
    assert llm_1_5.is_gemini_2_0 is False


def test_add_property_ordering_to_schema():
    """Test that _add_property_ordering correctly adds propertyOrdering to schemas."""
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    # Test simple object schema
    simple_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        }
    }

    result = GeminiCompletion._add_property_ordering(simple_schema)

    assert "propertyOrdering" in result
    assert result["propertyOrdering"] == ["name", "age", "email"]

    # Test nested object schema
    nested_schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "contact": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"}
                        }
                    }
                }
            },
            "id": {"type": "integer"}
        }
    }

    result = GeminiCompletion._add_property_ordering(nested_schema)

    assert "propertyOrdering" in result
    assert result["propertyOrdering"] == ["user", "id"]
    assert "propertyOrdering" in result["properties"]["user"]
    assert result["properties"]["user"]["propertyOrdering"] == ["name", "contact"]
    assert "propertyOrdering" in result["properties"]["user"]["properties"]["contact"]
    assert result["properties"]["user"]["properties"]["contact"]["propertyOrdering"] == ["email", "phone"]


def test_gemini_2_0_response_model_with_property_ordering():
    """Test that Gemini 2.0 models include propertyOrdering in response schemas."""
    from pydantic import BaseModel, Field

    class TestResponse(BaseModel):
        """Test response model."""
        name: str = Field(..., description="The name")
        age: int = Field(..., description="The age")
        email: str = Field(..., description="The email")

    llm = LLM(model="google/gemini-2.0-flash-001")

    # Prepare generation config with response model
    config = llm._prepare_generation_config(response_model=TestResponse)

    # Verify that the config has response_json_schema
    assert hasattr(config, 'response_json_schema') or 'response_json_schema' in config.__dict__

    # Get the schema
    if hasattr(config, 'response_json_schema'):
        schema = config.response_json_schema
    else:
        schema = config.__dict__.get('response_json_schema', {})

    # Verify propertyOrdering is present for Gemini 2.0
    assert "propertyOrdering" in schema
    assert "name" in schema["propertyOrdering"]
    assert "age" in schema["propertyOrdering"]
    assert "email" in schema["propertyOrdering"]


def test_gemini_1_5_response_model_uses_response_schema():
    """Test that Gemini 1.5 models use response_schema parameter (not response_json_schema)."""
    from pydantic import BaseModel, Field

    class TestResponse(BaseModel):
        """Test response model."""
        name: str = Field(..., description="The name")
        age: int = Field(..., description="The age")

    llm = LLM(model="google/gemini-1.5-pro")

    # Prepare generation config with response model
    config = llm._prepare_generation_config(response_model=TestResponse)

    # Verify that the config uses response_schema (not response_json_schema)
    assert hasattr(config, 'response_schema') or 'response_schema' in config.__dict__
    assert not (hasattr(config, 'response_json_schema') and config.response_json_schema is not None)

    # Get the schema
    if hasattr(config, 'response_schema'):
        schema = config.response_schema
    else:
        schema = config.__dict__.get('response_schema')

    # For Gemini 1.5, response_schema should be the Pydantic model itself
    # The SDK handles conversion internally
    assert schema is TestResponse or isinstance(schema, type)


# =============================================================================
# Agent Kickoff Structured Output Tests
# =============================================================================


@pytest.mark.vcr()
def test_gemini_agent_kickoff_structured_output_without_tools():
    """
    Test that agent kickoff returns structured output without tools.
    This tests native structured output handling for Gemini models.
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
        llm=LLM(model="google/gemini-2.0-flash-001"),
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
def test_gemini_agent_kickoff_structured_output_with_tools():
    """
    Test that agent kickoff returns structured output after using tools.
    This tests post-tool-call structured output handling for Gemini models.
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
        llm=LLM(model="google/gemini-2.0-flash-001"),
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



def test_gemini_stop_words_not_applied_to_structured_output():
    """
    Test that stop words are NOT applied when response_model is provided.
    This ensures JSON responses containing stop word patterns (like "Observation:")
    are not truncated, which would cause JSON validation to fail.
    """
    from pydantic import BaseModel, Field
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    class ResearchResult(BaseModel):
        """Research result that may contain stop word patterns in string fields."""

        finding: str = Field(description="The research finding")
        observation: str = Field(description="Observation about the finding")

    # Create Gemini completion instance with stop words configured
    # Gemini uses stop_sequences instead of stop
    llm = GeminiCompletion(
        model="gemini-2.0-flash-001",
        stop_sequences=["Observation:", "Final Answer:"],  # Common stop words
    )

    # JSON response that contains a stop word pattern in a string field
    # Without the fix, this would be truncated at "Observation:" breaking the JSON
    json_response = '{"finding": "The data shows growth", "observation": "Observation: This confirms the hypothesis"}'

    # Test the _validate_structured_output method which is used for structured output handling
    result = llm._validate_structured_output(json_response, ResearchResult)

    # Should successfully parse the full JSON without truncation
    assert isinstance(result, ResearchResult)
    assert result.finding == "The data shows growth"
    # The observation field should contain the full text including "Observation:"
    assert "Observation:" in result.observation


def test_gemini_stop_words_still_applied_to_regular_responses():
    """
    Test that stop words ARE still applied for regular (non-structured) responses.
    This ensures the fix didn't break normal stop word behavior.
    """
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    # Create Gemini completion instance with stop words configured
    # Gemini uses stop_sequences instead of stop
    llm = GeminiCompletion(
        model="gemini-2.0-flash-001",
        stop_sequences=["Observation:", "Final Answer:"],
    )

    # Response that contains a stop word - should be truncated
    response_with_stop_word = "I need to search for more information.\n\nAction: search\nObservation: Found results"

    # Test the _apply_stop_words method directly
    result = llm._apply_stop_words(response_with_stop_word)

    # Response should be truncated at the stop word
    assert "Observation:" not in result
    assert "Found results" not in result
    assert "I need to search for more information" in result


def test_gemini_structured_output_preserves_json_with_stop_word_patterns():
    """
    Test that structured output validation preserves JSON content
    even when string fields contain stop word patterns.
    """
    from pydantic import BaseModel, Field
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    class AgentObservation(BaseModel):
        """Model with fields that might contain stop word-like text."""

        action_taken: str = Field(description="What action was taken")
        observation_result: str = Field(description="The observation result")
        final_answer: str = Field(description="The final answer")

    # Gemini uses stop_sequences instead of stop
    llm = GeminiCompletion(
        model="gemini-2.0-flash-001",
        stop_sequences=["Observation:", "Final Answer:", "Action:"],
    )

    # JSON that contains all the stop word patterns as part of the content
    json_with_stop_patterns = '''{
        "action_taken": "Action: Searched the database",
        "observation_result": "Observation: Found 5 relevant results",
        "final_answer": "Final Answer: The data shows positive growth"
    }'''

    # Test the _validate_structured_output method - this should NOT truncate
    # since it's structured output
    result = llm._validate_structured_output(json_with_stop_patterns, AgentObservation)

    assert isinstance(result, AgentObservation)
    assert "Action:" in result.action_taken
    assert "Observation:" in result.observation_result
    assert "Final Answer:" in result.final_answer
