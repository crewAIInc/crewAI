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
def mock_aws_credentials():
    """Automatically mock AWS credentials and boto3 Session for all tests in this module."""
    with patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_DEFAULT_REGION": "us-east-1"
    }):
        # Mock boto3 Session to prevent actual AWS connections
        with patch('crewai.llms.providers.bedrock.completion.Session') as mock_session_class:
            # Create mock session instance
            mock_session_instance = MagicMock()
            mock_client = MagicMock()

            # Set up default mock responses to prevent hanging
            default_response = {
                'output': {
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {'text': 'Test response'}
                        ]
                    }
                },
                'usage': {
                    'inputTokens': 10,
                    'outputTokens': 5,
                    'totalTokens': 15
                }
            }
            mock_client.converse.return_value = default_response
            mock_client.converse_stream.return_value = {'stream': []}

            # Configure the mock session instance to return the mock client
            mock_session_instance.client.return_value = mock_client

            # Configure the mock Session class to return the mock session instance
            mock_session_class.return_value = mock_session_instance

            yield mock_session_class, mock_client


def test_bedrock_completion_is_used_when_bedrock_provider():
    """
    Test that BedrockCompletion from completion.py is used when LLM uses provider 'bedrock'
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    assert llm.__class__.__name__ == "BedrockCompletion"
    assert llm.provider == "bedrock"
    assert llm.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_bedrock_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using Bedrock provider
    """
    module_name = "crewai.llms.providers.bedrock.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'BedrockCompletion')


def test_native_bedrock_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native Bedrock completion fails.

    With the new behavior, when a native provider is in SUPPORTED_NATIVE_PROVIDERS
    but fails to instantiate, we raise an ImportError instead of silently falling back.
    This provides clearer error messages to users about missing dependencies.
    """
    # Mock the _get_native_provider to return a failing class
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native AWS Bedrock SDK failed")

        mock_get_provider.return_value = FailingCompletion

        # This should raise ImportError with clear message
        with pytest.raises(ImportError) as excinfo:
            LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

        # Verify the error message is helpful
        assert "Error importing native provider" in str(excinfo.value)
        assert "Native AWS Bedrock SDK failed" in str(excinfo.value)


def test_bedrock_completion_initialization_parameters():
    """
    Test that BedrockCompletion is initialized with correct parameters
    """
    llm = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9,
        top_k=40,
        region_name="us-west-2"
    )

    from crewai.llms.providers.bedrock.completion import BedrockCompletion
    assert isinstance(llm, BedrockCompletion)
    assert llm.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2000
    assert llm.top_p == 0.9
    assert llm.top_k == 40
    assert llm.region_name == "us-west-2"


def test_bedrock_specific_parameters():
    """
    Test Bedrock-specific parameters like stop_sequences and streaming
    """
    llm = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        stop_sequences=["Human:", "Assistant:"],
        stream=True,
        region_name="us-east-1"
    )

    from crewai.llms.providers.bedrock.completion import BedrockCompletion
    assert isinstance(llm, BedrockCompletion)
    assert llm.stop_sequences == ["Human:", "Assistant:"]
    assert llm.stream == True
    assert llm.region_name == "us-east-1"


def test_bedrock_completion_call():
    """
    Test that BedrockCompletion call method works
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Mock the call method on the instance
    with patch.object(llm, 'call', return_value="Hello! I'm Claude on Bedrock, ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm Claude on Bedrock, ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


def test_bedrock_completion_called_during_crew_execution():
    """
    Test that BedrockCompletion.call is actually invoked when running a crew
    """
    # Create the LLM instance first
    bedrock_llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Mock the call method on the specific instance
    with patch.object(bedrock_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:

        # Create agent with explicit LLM configuration
        agent = Agent(
            role="Research Assistant",
            goal="Find population info",
            backstory="You research populations.",
            llm=bedrock_llm,
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


@pytest.mark.skip(reason="Crew execution test - may hang, needs investigation")
def test_bedrock_completion_call_arguments():
    """
    Test that BedrockCompletion.call is invoked with correct arguments
    """
    # Create LLM instance first
    bedrock_llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Mock the instance method
    with patch.object(bedrock_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed successfully."

        agent = Agent(
            role="Test Agent",
            goal="Complete a simple task",
            backstory="You are a test agent.",
            llm=bedrock_llm  # Use same instance
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


def test_multiple_bedrock_calls_in_crew():
    """
    Test that BedrockCompletion.call is invoked multiple times for multiple tasks
    """
    # Create LLM instance first
    bedrock_llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Mock the instance method
    with patch.object(bedrock_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent = Agent(
            role="Multi-task Agent",
            goal="Complete multiple tasks",
            backstory="You can handle multiple tasks.",
            llm=bedrock_llm  # Use same instance
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

def test_bedrock_completion_with_tools():
    """
    Test that BedrockCompletion.call is invoked with tools when agent has tools
    """
    from crewai.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing"""
        return f"Tool result for: {query}"

    # Create LLM instance first
    bedrock_llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Mock the instance method
    with patch.object(bedrock_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed with tools."

        agent = Agent(
            role="Tool User",
            goal="Use tools to complete tasks",
            backstory="You can use tools.",
            llm=bedrock_llm,  # Use same instance
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


def test_bedrock_raises_error_when_model_not_found(mock_aws_credentials):
    """Test that BedrockCompletion raises appropriate error when model not found"""
    from botocore.exceptions import ClientError

    # Get the mock client from the fixture
    _, mock_client = mock_aws_credentials

    error_response = {
        'Error': {
            'Code': 'ResourceNotFoundException',
            'Message': 'Could not resolve the foundation model from the model identifier'
        }
    }
    mock_client.converse.side_effect = ClientError(error_response, 'converse')

    llm = LLM(model="bedrock/model-doesnt-exist")

    with pytest.raises(Exception):  # Should raise some error for unsupported model
        llm.call("Hello")


def test_bedrock_aws_credentials_configuration():
    """
    Test that AWS credentials configuration works properly
    """
    # Test with environment variables
    with patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_DEFAULT_REGION": "us-east-1"
    }):
        llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

        from crewai.llms.providers.bedrock.completion import BedrockCompletion
        assert isinstance(llm, BedrockCompletion)
        assert llm.region_name == "us-east-1"

    # Test with explicit credentials
    llm_explicit = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_access_key_id="explicit-key",
        aws_secret_access_key="explicit-secret",
        region_name="us-west-2"
    )
    assert isinstance(llm_explicit, BedrockCompletion)
    assert llm_explicit.region_name == "us-west-2"


def test_bedrock_model_capabilities():
    """
    Test that model capabilities are correctly identified
    """
    # Test Claude model
    llm_claude = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    from crewai.llms.providers.bedrock.completion import BedrockCompletion
    assert isinstance(llm_claude, BedrockCompletion)
    assert llm_claude.is_claude_model == True
    assert llm_claude.supports_tools == True

    # Test other Bedrock model
    llm_titan = LLM(model="bedrock/amazon.titan-text-express-v1")
    assert isinstance(llm_titan, BedrockCompletion)
    assert llm_titan.supports_tools == True


def test_bedrock_inference_config():
    """
    Test that inference config is properly prepared
    """
    llm = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_tokens=1000
    )

    from crewai.llms.providers.bedrock.completion import BedrockCompletion
    assert isinstance(llm, BedrockCompletion)

    # Test config preparation
    config = llm._get_inference_config()

    # Verify config has the expected parameters
    assert 'temperature' in config
    assert config['temperature'] == 0.7
    assert 'topP' in config
    assert config['topP'] == 0.9
    assert 'maxTokens' in config
    assert config['maxTokens'] == 1000
    assert 'topK' in config
    assert config['topK'] == 40


def test_bedrock_model_detection():
    """
    Test that various Bedrock model formats are properly detected
    """
    # Test Bedrock model naming patterns
    bedrock_test_cases = [
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/amazon.titan-text-express-v1",
        "bedrock/meta.llama3-70b-instruct-v1:0"
    ]

    for model_name in bedrock_test_cases:
        llm = LLM(model=model_name)
        from crewai.llms.providers.bedrock.completion import BedrockCompletion
        assert isinstance(llm, BedrockCompletion), f"Failed for model: {model_name}"


def test_bedrock_supports_stop_words():
    """
    Test that Bedrock models support stop sequences
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert llm.supports_stop_words() == True


def test_bedrock_context_window_size():
    """
    Test that Bedrock models return correct context window sizes
    """
    # Test Claude 3.5 Sonnet
    llm_claude = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    context_size_claude = llm_claude.get_context_window_size()
    assert context_size_claude > 150000  # Should be substantial (200K tokens with ratio)

    # Test Titan
    llm_titan = LLM(model="bedrock/amazon.titan-text-express-v1")
    context_size_titan = llm_titan.get_context_window_size()
    assert context_size_titan > 5000  # Should have 8K context window


def test_bedrock_message_formatting():
    """
    Test that messages are properly formatted for Bedrock Converse API
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Test message formatting
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    formatted_messages, system_message = llm._format_messages_for_converse(test_messages)

    # System message should be extracted
    assert system_message == "You are a helpful assistant."

    # Remaining messages should be in Converse format
    assert len(formatted_messages) >= 3  # Should have user, assistant, user messages

    # First message should be user role
    assert formatted_messages[0]["role"] == "user"
    # Second should be assistant
    assert formatted_messages[1]["role"] == "assistant"

    # Messages should have content array with text
    assert isinstance(formatted_messages[0]["content"], list)
    assert "text" in formatted_messages[0]["content"][0]


def test_bedrock_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    # Test non-streaming
    llm_no_stream = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", stream=False)
    assert llm_no_stream.stream == False

    # Test streaming
    llm_stream = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", stream=True)
    assert llm_stream.stream == True


def test_bedrock_tool_conversion():
    """
    Test that tools are properly converted to Bedrock Converse format
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

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
    bedrock_tools = llm._format_tools_for_converse(crewai_tools)

    assert len(bedrock_tools) == 1
    # Bedrock tools should have toolSpec structure
    assert "toolSpec" in bedrock_tools[0]
    assert bedrock_tools[0]["toolSpec"]["name"] == "test_tool"
    assert bedrock_tools[0]["toolSpec"]["description"] == "A test tool"
    assert "inputSchema" in bedrock_tools[0]["toolSpec"]


def test_bedrock_environment_variable_credentials(mock_aws_credentials):
    """
    Test that AWS credentials are properly loaded from environment
    """
    mock_session_class, _ = mock_aws_credentials

    # Reset the mock to clear any previous calls
    mock_session_class.reset_mock()

    with patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "test-access-key-123",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key-456"
    }):
        llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

        # Verify Session was called with environment credentials
        assert mock_session_class.called
        # Get the most recent call - Session is called as Session(...)
        call_kwargs = mock_session_class.call_args[1] if mock_session_class.call_args else {}
        assert call_kwargs.get('aws_access_key_id') == "test-access-key-123"
        assert call_kwargs.get('aws_secret_access_key') == "test-secret-key-456"


def test_bedrock_token_usage_tracking():
    """
    Test that token usage is properly tracked for Bedrock responses
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Mock the Bedrock response with usage information
    with patch.object(llm.client, 'converse') as mock_converse:
        mock_response = {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'text': 'test response'}
                    ]
                }
            },
            'usage': {
                'inputTokens': 50,
                'outputTokens': 25,
                'totalTokens': 75
            }
        }
        mock_converse.return_value = mock_response

        result = llm.call("Hello")

        # Verify the response
        assert result == "test response"

        # Verify token usage was tracked
        assert llm._token_usage['prompt_tokens'] == 50
        assert llm._token_usage['completion_tokens'] == 25
        assert llm._token_usage['total_tokens'] == 75


def test_bedrock_tool_use_conversation_flow():
    """
    Test that the Bedrock completion properly handles tool use conversation flow
    """
    from unittest.mock import Mock

    # Create BedrockCompletion instance
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Mock tool function
    def mock_weather_tool(location: str) -> str:
        return f"The weather in {location} is sunny and 75°F"

    available_functions = {"get_weather": mock_weather_tool}

    # Mock the Bedrock client responses
    with patch.object(llm.client, 'converse') as mock_converse:
        # First response: tool use request
        tool_use_response = {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [
                        {
                            'toolUse': {
                                'toolUseId': 'tool-123',
                                'name': 'get_weather',
                                'input': {'location': 'San Francisco'}
                            }
                        }
                    ]
                }
            },
            'usage': {
                'inputTokens': 100,
                'outputTokens': 50,
                'totalTokens': 150
            }
        }

        # Second response: final answer after tool execution
        final_response = {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'text': 'Based on the weather data, it is sunny and 75°F in San Francisco.'}
                    ]
                }
            },
            'usage': {
                'inputTokens': 120,
                'outputTokens': 30,
                'totalTokens': 150
            }
        }

        # Configure mock to return different responses on successive calls
        mock_converse.side_effect = [tool_use_response, final_response]

        # Test the call
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        result = llm.call(
            messages=messages,
            available_functions=available_functions
        )

        # Verify the final response contains the weather information
        assert "sunny" in result.lower() or "75" in result

        # Verify that the API was called twice (once for tool use, once for final answer)
        assert mock_converse.call_count == 2


def test_bedrock_handles_cohere_conversation_requirements():
    """
    Test that Bedrock properly handles Cohere model's requirement for user message at end
    """
    llm = LLM(model="bedrock/cohere.command-r-plus-v1:0")

    # Test message formatting with conversation ending in assistant message
    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    formatted_messages, system_message = llm._format_messages_for_converse(test_messages)

    # For Cohere models, should add a user message at the end
    assert formatted_messages[-1]["role"] == "user"
    assert "continue" in formatted_messages[-1]["content"][0]["text"].lower()


def test_bedrock_client_error_handling():
    """
    Test that Bedrock properly handles various AWS client errors
    """
    from botocore.exceptions import ClientError

    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Test ValidationException
    with patch.object(llm.client, 'converse') as mock_converse:
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid request format'
            }
        }
        mock_converse.side_effect = ClientError(error_response, 'converse')

        with pytest.raises(ValueError) as exc_info:
            llm.call("Hello")
        assert "validation" in str(exc_info.value).lower()

    # Test ThrottlingException
    with patch.object(llm.client, 'converse') as mock_converse:
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate limit exceeded'
            }
        }
        mock_converse.side_effect = ClientError(error_response, 'converse')

        with pytest.raises(RuntimeError) as exc_info:
            llm.call("Hello")
        assert "throttled" in str(exc_info.value).lower()


def test_bedrock_stop_sequences_sync():
    """Test that stop and stop_sequences attributes stay synchronized."""
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Test setting stop as a list
    llm.stop = ["\nObservation:", "\nThought:"]
    assert list(llm.stop_sequences) == ["\nObservation:", "\nThought:"]
    assert llm.stop == ["\nObservation:", "\nThought:"]

    # Test setting stop as a string
    llm.stop = "\nFinal Answer:"
    assert list(llm.stop_sequences) == ["\nFinal Answer:"]
    assert llm.stop == ["\nFinal Answer:"]

    # Test setting stop as None
    llm.stop = None
    assert list(llm.stop_sequences) == []
    assert llm.stop == []


def test_bedrock_stop_sequences_sent_to_api():
    """Test that stop_sequences are properly sent to the Bedrock API."""
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Set stop sequences via the stop attribute (simulating CrewAgentExecutor)
    llm.stop = ["\nObservation:", "\nThought:"]

    # Patch the API call to capture parameters without making real call
    with patch.object(llm.client, 'converse') as mock_converse:
        mock_response = {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [{'text': 'Hello'}]
                }
            },
            'usage': {
                'inputTokens': 10,
                'outputTokens': 5,
                'totalTokens': 15
            }
        }
        mock_converse.return_value = mock_response

        llm.call("Say hello in one word")

        # Verify stop_sequences were passed to the API in the inference config
        call_kwargs = mock_converse.call_args[1]
        assert "inferenceConfig" in call_kwargs
        assert "stopSequences" in call_kwargs["inferenceConfig"]
        assert call_kwargs["inferenceConfig"]["stopSequences"] == ["\nObservation:", "\nThought:"]
