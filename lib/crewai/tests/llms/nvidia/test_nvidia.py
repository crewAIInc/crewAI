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
def mock_nvidia_api_key():
    """Automatically mock NVIDIA_API_KEY for all tests in this module."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        yield


def test_nvidia_completion_is_used_when_nvidia_provider():
    """
    Test that NvidiaCompletion from completion.py is used when LLM uses provider 'nvidia'
    """
    llm = LLM(model="nvidia/llama-3.1-70b-instruct")

    assert llm.__class__.__name__ == "NvidiaCompletion"
    assert llm.provider == "nvidia"
    assert llm.model == "llama-3.1-70b-instruct"


def test_nvidia_completion_is_used_when_model_has_slash():
    """
    Test that NvidiaCompletion is used when model contains '/' and NVIDIA_API_KEY is set
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(model="meta/llama-3.1-70b-instruct")

        from crewai.llms.providers.nvidia.completion import NvidiaCompletion
        assert isinstance(llm, NvidiaCompletion)
        assert llm.provider == "nvidia"
        assert llm.model == "meta/llama-3.1-70b-instruct"


def test_nvidia_falls_back_when_no_api_key():
    """
    Test that NVIDIA models fall back to LiteLLM when no NVIDIA_API_KEY is set
    """
    # Ensure no NVIDIA API key
    with patch.dict(os.environ, {}, clear=True):
        llm = LLM(model="meta/llama-3.1-70b-instruct")

        # Should not be NvidiaCompletion
        assert llm.__class__.__name__ != "NvidiaCompletion"


def test_nvidia_tool_use_conversation_flow():
    """
    Test that the NVIDIA completion properly handles tool use conversation flow
    """
    from unittest.mock import Mock, patch
    from crewai.llms.providers.nvidia.completion import NvidiaCompletion

    # Create NvidiaCompletion instance
    completion = NvidiaCompletion(model="meta/llama-3.1-70b-instruct")

    # Mock tool function
    def mock_weather_tool(location: str) -> str:
        return f"The weather in {location} is sunny and 75°F"

    available_functions = {"get_weather": mock_weather_tool}

    # Mock the OpenAI client responses
    with patch.object(completion.client.chat.completions, 'create') as mock_create:
        # Mock function call in response
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.arguments = '{"location": "San Francisco"}'

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function_call

        mock_choice = Mock()
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.message.content = None

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_create.return_value = mock_response

        # Test the call
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        result = completion.call(
            messages=messages,
            available_functions=available_functions
        )

        # Verify the tool was executed and returned the result
        assert result == "The weather in San Francisco is sunny and 75°F"

        # Verify that the API was called
        assert mock_create.called


def test_nvidia_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using NVIDIA provider
    """
    module_name = "crewai.llms.providers.nvidia.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        LLM(model="nvidia/llama-3.1-70b-instruct")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'NvidiaCompletion')


def test_native_nvidia_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native NVIDIA completion fails.

    With the new behavior, when a native provider is in SUPPORTED_NATIVE_PROVIDERS
    but fails to instantiate, we raise an ImportError instead of silently falling back.
    This provides clearer error messages to users about missing dependencies.
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native NVIDIA SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should raise ImportError with clear message
        with pytest.raises(ImportError) as excinfo:
            with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
                LLM(model="nvidia/llama-3.1-70b-instruct")

        # Verify the error message is helpful
        assert "Error importing native provider" in str(excinfo.value)
        assert "Native NVIDIA SDK failed" in str(excinfo.value)


def test_nvidia_completion_initialization_parameters():
    """
    Test that NvidiaCompletion is initialized with correct parameters
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(
            model="nvidia/llama-3.1-70b-instruct",
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.1,
            api_key="test-key"
        )

        from crewai.llms.providers.nvidia.completion import NvidiaCompletion
        assert isinstance(llm, NvidiaCompletion)
        assert llm.model == "llama-3.1-70b-instruct"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2000
        assert llm.top_p == 0.9
        assert llm.frequency_penalty == 0.1


def test_nvidia_specific_parameters():
    """
    Test NVIDIA-specific parameters like seed, stream, and response_format
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(
            model="nvidia/llama-3.1-70b-instruct",
            seed=42,
            stream=True,
            response_format={"type": "json_object"},
            logprobs=True,
            top_logprobs=5
        )

        from crewai.llms.providers.nvidia.completion import NvidiaCompletion
        assert isinstance(llm, NvidiaCompletion)
        assert llm.seed == 42
        assert llm.stream == True
        assert llm.response_format == {"type": "json_object"}
        assert llm.logprobs == True
        assert llm.top_logprobs == 5


def test_nvidia_completion_call():
    """
    Test that NvidiaCompletion call method works
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        # Mock the call method on the instance
        with patch.object(llm, 'call', return_value="Hello! I'm NVIDIA Llama, ready to help.") as mock_call:
            result = llm.call("Hello, how are you?")

            assert result == "Hello! I'm NVIDIA Llama, ready to help."
            mock_call.assert_called_once_with("Hello, how are you?")


def test_nvidia_completion_called_during_crew_execution():
    """
    Test that NvidiaCompletion.call is actually invoked when running a crew
    """
    # Create the LLM instance first
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        nvidia_llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        # Mock the call method on the specific instance
        with patch.object(nvidia_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:

            # Create agent with explicit LLM configuration
            agent = Agent(
                role="Research Assistant",
                goal="Find population info",
                backstory="You research populations.",
                llm=nvidia_llm,
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


def test_nvidia_completion_call_arguments():
    """
    Test that NvidiaCompletion.call is invoked with correct arguments
    """
    # Create LLM instance first
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        nvidia_llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        # Mock the instance method
        with patch.object(nvidia_llm, 'call') as mock_call:
            mock_call.return_value = "Task completed successfully."

            agent = Agent(
                role="Test Agent",
                goal="Complete a simple task",
                backstory="You are a test agent.",
                llm=nvidia_llm  # Use same instance
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


def test_multiple_nvidia_calls_in_crew():
    """
    Test that NvidiaCompletion.call is invoked multiple times for multiple tasks
    """
    # Create LLM instance first
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        nvidia_llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        # Mock the instance method
        with patch.object(nvidia_llm, 'call') as mock_call:
            mock_call.return_value = "Task completed."

            agent = Agent(
                role="Multi-task Agent",
                goal="Complete multiple tasks",
                backstory="You can handle multiple tasks.",
                llm=nvidia_llm  # Use same instance
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


def test_nvidia_completion_with_tools():
    """
    Test that NvidiaCompletion.call is invoked with tools when agent has tools
    """
    from crewai.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing"""
        return f"Tool result for: {query}"

    # Create LLM instance first
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        nvidia_llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        # Mock the instance method
        with patch.object(nvidia_llm, 'call') as mock_call:
            mock_call.return_value = "Task completed with tools."

            agent = Agent(
                role="Tool User",
                goal="Use tools to complete tasks",
                backstory="You can use tools.",
                llm=nvidia_llm,  # Use same instance
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


def test_nvidia_raises_error_when_model_not_supported():
    """Test that NvidiaCompletion raises ValueError when model not supported"""

    # Mock the OpenAI client to raise an error
    with patch('crewai.llms.providers.nvidia.completion.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 404

        from openai import NotFoundError
        mock_client.chat.completions.create.side_effect = NotFoundError("Model not found", response=mock_response, body=None)

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            llm = LLM(model="nvidia/model-doesnt-exist")

            with pytest.raises(ValueError):  # Should raise ValueError for unsupported model
                llm.call("Hello")


def test_nvidia_api_key_configuration():
    """
    Test that API key configuration works for both NVIDIA_API_KEY and NVIDIA_NIM_API_KEY
    """
    # Test with NVIDIA_API_KEY
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-nvidia-key"}):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        from crewai.llms.providers.nvidia.completion import NvidiaCompletion
        assert isinstance(llm, NvidiaCompletion)
        assert llm.api_key == "test-nvidia-key"

    # Test with NVIDIA_NIM_API_KEY
    with patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "test-nim-key"}, clear=True):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        assert isinstance(llm, NvidiaCompletion)
        assert llm.api_key == "test-nim-key"


def test_nvidia_model_capabilities():
    """
    Test that model capabilities are correctly identified
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        # Test Llama 3.1 model
        llm_llama = LLM(model="meta/llama-3.1-70b-instruct")
        from crewai.llms.providers.nvidia.completion import NvidiaCompletion
        assert isinstance(llm_llama, NvidiaCompletion)
        assert llm_llama.supports_tools == True

        # Test vision model
        llm_vision = LLM(model="meta/llama-3.2-90b-vision-instruct")
        assert isinstance(llm_vision, NvidiaCompletion)
        assert llm_vision.is_vision_model == True


def test_nvidia_generation_config():
    """
    Test that generation config is properly prepared
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(
            model="nvidia/llama-3.1-70b-instruct",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            max_tokens=1000
        )

        from crewai.llms.providers.nvidia.completion import NvidiaCompletion
        assert isinstance(llm, NvidiaCompletion)

        # Test config preparation
        params = llm._prepare_completion_params([])

        # Verify config has the expected parameters
        assert "temperature" in params
        assert params["temperature"] == 0.7
        assert "top_p" in params
        assert params["top_p"] == 0.9
        assert "frequency_penalty" in params
        assert params["frequency_penalty"] == 0.1
        assert "max_tokens" in params
        assert params["max_tokens"] == 1000


def test_nvidia_model_detection():
    """
    Test that various NVIDIA model formats are properly detected
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        # Test NVIDIA model naming patterns that actually work with provider detection
        nvidia_test_cases = [
            "nvidia/llama-3.1-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "qwen/qwen3-next-80b-a3b-instruct",
            "deepseek-ai/deepseek-r1",
            "google/gemma-2-27b-it",
            "mistralai/mistral-large-3-675b-instruct-2512"
        ]

        for model_name in nvidia_test_cases:
            llm = LLM(model=model_name)
            from crewai.llms.providers.nvidia.completion import NvidiaCompletion
            assert isinstance(llm, NvidiaCompletion), f"Failed for model: {model_name}"


def test_nvidia_supports_stop_words():
    """
    Test that NVIDIA models support stop sequences
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")
        assert llm.supports_stop_words() == True


def test_nvidia_context_window_size():
    """
    Test that NVIDIA models return correct context window sizes
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        # Test Llama 3.1 model
        llm_llama = LLM(model="meta/llama-3.1-70b-instruct")
        context_size_llama = llm_llama.get_context_window_size()
        assert context_size_llama > 100000  # Should be substantial

        # Test DeepSeek R1 model
        llm_deepseek = LLM(model="deepseek-ai/deepseek-r1")
        context_size_deepseek = llm_deepseek.get_context_window_size()
        assert context_size_deepseek > 100000  # Should be large


def test_nvidia_message_formatting():
    """
    Test that messages are properly formatted for NVIDIA API
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        # Test message formatting
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        formatted_messages = llm._format_messages(test_messages)

        # Should have all messages
        assert len(formatted_messages) == 4

        # Check roles are preserved
        assert formatted_messages[0]["role"] == "system"
        assert formatted_messages[1]["role"] == "user"
        assert formatted_messages[2]["role"] == "assistant"
        assert formatted_messages[3]["role"] == "user"


def test_nvidia_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        # Test non-streaming
        llm_no_stream = LLM(model="nvidia/llama-3.1-70b-instruct", stream=False)
        assert llm_no_stream.stream == False

        # Test streaming
        llm_stream = LLM(model="nvidia/llama-3.1-70b-instruct", stream=True)
        assert llm_stream.stream == True


def test_nvidia_tool_conversion():
    """
    Test that tools are properly converted to OpenAI format for NVIDIA
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")

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
        nvidia_tools = llm._convert_tools_for_interference(crewai_tools)

        assert len(nvidia_tools) == 1
        # NVIDIA tools are in OpenAI format
        assert nvidia_tools[0]["type"] == "function"
        assert nvidia_tools[0]["function"]["name"] == "test_tool"
        assert nvidia_tools[0]["function"]["description"] == "A test tool"


def test_nvidia_environment_variable_api_key():
    """
    Test that NVIDIA API key is properly loaded from environment
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-nvidia-key"}):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        assert llm.client is not None
        assert hasattr(llm.client, 'chat')
        assert llm.api_key == "test-nvidia-key"


def test_nvidia_token_usage_tracking():
    """
    Test that token usage is properly tracked for NVIDIA responses
    """
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(model="nvidia/llama-3.1-70b-instruct")

        # Mock the OpenAI response with usage information
        with patch.object(llm.client.chat.completions, 'create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test response"
            mock_response.usage = MagicMock(
                prompt_tokens=50,
                completion_tokens=25,
                total_tokens=75
            )
            mock_create.return_value = mock_response

            result = llm.call("Hello")

            # Verify the response
            assert result == "test response"

            # Verify token usage was extracted
            usage = llm._extract_token_usage(mock_response)
            assert usage["prompt_tokens"] == 50
            assert usage["completion_tokens"] == 25
            assert usage["total_tokens"] == 75


def test_nvidia_reasoning_model_detection():
    """Test that reasoning models like DeepSeek R1 are properly detected."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        llm = LLM(model="deepseek-ai/deepseek-r1")

        from crewai.llms.providers.nvidia.completion import NvidiaCompletion
        assert isinstance(llm, NvidiaCompletion)

        # Test that reasoning models get default max_tokens when not specified
        config = llm._prepare_completion_params([])
        assert "max_tokens" in config
        assert config["max_tokens"] == 4096  # Default for reasoning models


def test_nvidia_vision_model_detection():
    """Test that vision models are properly detected."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        vision_models = [
            "meta/llama-3.2-90b-vision-instruct",
            "meta/llama-3.2-11b-vision-instruct",
            "microsoft/phi-3-vision-128k-instruct"
        ]

        for model in vision_models:
            llm = LLM(model=model)
            from crewai.llms.providers.nvidia.completion import NvidiaCompletion
            assert isinstance(llm, NvidiaCompletion)
            assert llm.is_vision_model == True


@pytest.mark.vcr()
@pytest.mark.skip(reason="VCR cannot replay SSE streaming responses")
def test_nvidia_streaming_returns_usage_metrics():
    """
    Test that NVIDIA streaming calls return proper token usage metrics.
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the capital of Japan",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="meta/llama-3.1-70b-instruct", stream=True),
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