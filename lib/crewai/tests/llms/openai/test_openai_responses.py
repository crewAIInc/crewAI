"""Tests for OpenAI Responses API integration."""

import json
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.openai.responses import OpenAIResponsesCompletion


class TestOpenAIResponsesProviderRouting:
    """Tests for provider routing to OpenAIResponsesCompletion."""

    def test_openai_responses_completion_is_used_when_provider_specified(self):
        """Test that OpenAIResponsesCompletion is used when provider='openai_responses'."""
        llm = LLM(model="gpt-4o", provider="openai_responses")

        assert isinstance(llm, OpenAIResponsesCompletion)
        assert llm.provider == "openai_responses"
        assert llm.model == "gpt-4o"

    def test_openai_responses_completion_is_used_with_prefix(self):
        """Test that OpenAIResponsesCompletion is used with openai_responses/ prefix."""
        llm = LLM(model="openai_responses/gpt-4o")

        assert isinstance(llm, OpenAIResponsesCompletion)
        assert llm.provider == "openai_responses"
        assert llm.model == "gpt-4o"

    def test_openai_responses_completion_initialization_parameters(self):
        """Test that OpenAIResponsesCompletion is initialized with correct parameters."""
        llm = LLM(
            model="gpt-4o",
            provider="openai_responses",
            temperature=0.7,
            max_output_tokens=1000,
            api_key="test-key",
        )

        assert isinstance(llm, OpenAIResponsesCompletion)
        assert llm.model == "gpt-4o"
        assert llm.temperature == 0.7
        assert llm.max_output_tokens == 1000

    def test_openai_responses_with_reasoning_effort(self):
        """Test that reasoning_effort parameter is accepted for o-series models."""
        llm = LLM(
            model="o3-mini",
            provider="openai_responses",
            reasoning_effort="high",
        )

        assert isinstance(llm, OpenAIResponsesCompletion)
        assert llm.reasoning_effort == "high"
        assert llm.is_o_model is True

    def test_openai_responses_with_previous_response_id(self):
        """Test that previous_response_id parameter is accepted."""
        llm = LLM(
            model="gpt-4o",
            provider="openai_responses",
            previous_response_id="resp_12345",
        )

        assert isinstance(llm, OpenAIResponsesCompletion)
        assert llm.previous_response_id == "resp_12345"

    def test_openai_responses_with_store_parameter(self):
        """Test that store parameter is accepted."""
        llm = LLM(
            model="gpt-4o",
            provider="openai_responses",
            store=True,
        )

        assert isinstance(llm, OpenAIResponsesCompletion)
        assert llm.store is True


class TestOpenAIResponsesMessageConversion:
    """Tests for message conversion to Responses API format."""

    def test_convert_simple_user_message(self):
        """Test conversion of a simple user message."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        messages = [{"role": "user", "content": "Hello, world!"}]

        instructions, input_content = llm._convert_messages_to_responses_format(
            messages
        )

        assert instructions is None
        assert input_content == "Hello, world!"

    def test_convert_system_message_to_instructions(self):
        """Test that system messages are converted to instructions."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        instructions, input_content = llm._convert_messages_to_responses_format(
            messages
        )

        assert instructions == "You are a helpful assistant."
        assert input_content == "Hello!"

    def test_convert_multiple_system_messages(self):
        """Test that multiple system messages are concatenated."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello!"},
        ]

        instructions, input_content = llm._convert_messages_to_responses_format(
            messages
        )

        assert instructions == "You are a helpful assistant.\n\nBe concise."
        assert input_content == "Hello!"

    def test_convert_multi_turn_conversation(self):
        """Test conversion of multi-turn conversation."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        instructions, input_content = llm._convert_messages_to_responses_format(
            messages
        )

        assert instructions == "You are a helpful assistant."
        assert isinstance(input_content, list)
        assert len(input_content) == 3
        assert input_content[0]["role"] == "user"
        assert input_content[0]["content"] == "Hello!"
        assert input_content[1]["role"] == "assistant"
        assert input_content[1]["content"] == "Hi there!"
        assert input_content[2]["role"] == "user"
        assert input_content[2]["content"] == "How are you?"


class TestOpenAIResponsesToolConversion:
    """Tests for tool conversion to Responses API format."""

    def test_convert_tools_for_responses(self):
        """Test conversion of CrewAI tools to Responses API format."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")

        tools = [
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        with patch(
            "crewai.llms.providers.utils.common.safe_tool_conversion"
        ) as mock_convert:
            mock_convert.return_value = (
                "search",
                "Search for information",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )

            responses_tools = llm._convert_tools_for_responses(tools)

            assert len(responses_tools) == 1
            assert responses_tools[0]["type"] == "function"
            assert responses_tools[0]["name"] == "search"
            assert responses_tools[0]["description"] == "Search for information"
            assert responses_tools[0]["strict"] is True


class TestOpenAIResponsesCall:
    """Tests for the call method."""

    def test_call_returns_response_text(self):
        """Test that call returns response text."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")

        mock_response = MagicMock()
        mock_response.id = "resp_12345"
        mock_response.output_text = "Hello! I'm ready to help."
        mock_response.output = []
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=20, total_tokens=30
        )

        with patch.object(llm.client.responses, "create", return_value=mock_response):
            result = llm.call("Hello, how are you?")

            assert result == "Hello! I'm ready to help."
            assert llm.last_response_id == "resp_12345"

    def test_call_with_tools_executes_function(self):
        """Test that call executes function when tool is called."""
        from openai.types.responses.response_function_tool_call import (
            ResponseFunctionToolCall,
        )

        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")

        mock_tool_call = ResponseFunctionToolCall(
            id="call_123",
            call_id="call_123",
            name="search",
            arguments='{"query": "test"}',
            type="function_call",
            status="completed",
        )

        mock_response = MagicMock()
        mock_response.id = "resp_12345"
        mock_response.output_text = ""
        mock_response.output = [mock_tool_call]
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=20, total_tokens=30
        )

        def search_function(query: str) -> str:
            return f"Results for: {query}"

        with patch.object(llm.client.responses, "create", return_value=mock_response):
            with patch.object(
                llm, "_handle_tool_execution", return_value="Results for: test"
            ) as mock_exec:
                result = llm.call(
                    "Search for test",
                    available_functions={"search": search_function},
                )
                mock_exec.assert_called_once()

    def test_call_tracks_token_usage(self):
        """Test that call tracks token usage."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")

        mock_response = MagicMock()
        mock_response.id = "resp_12345"
        mock_response.output_text = "Response"
        mock_response.output = []
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=20, total_tokens=30
        )

        with patch.object(llm.client.responses, "create", return_value=mock_response):
            llm.call("Hello")

            usage = llm.get_token_usage_summary()
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30


class TestOpenAIResponsesParamsPreparation:
    """Tests for parameter preparation."""

    def test_prepare_response_params_basic(self):
        """Test basic parameter preparation."""
        llm = OpenAIResponsesCompletion(
            model="gpt-4o", api_key="test-key", temperature=0.7
        )
        messages = [{"role": "user", "content": "Hello"}]

        params = llm._prepare_response_params(messages)

        assert params["model"] == "gpt-4o"
        assert params["input"] == "Hello"
        assert params["temperature"] == 0.7

    def test_prepare_response_params_with_reasoning_effort(self):
        """Test parameter preparation with reasoning effort for o-series models."""
        llm = OpenAIResponsesCompletion(
            model="o3-mini", api_key="test-key", reasoning_effort="high"
        )
        messages = [{"role": "user", "content": "Hello"}]

        params = llm._prepare_response_params(messages)

        assert params["model"] == "o3-mini"
        assert params["reasoning"] == {"effort": "high"}

    def test_prepare_response_params_with_previous_response_id(self):
        """Test parameter preparation with previous_response_id."""
        llm = OpenAIResponsesCompletion(
            model="gpt-4o", api_key="test-key", previous_response_id="resp_12345"
        )
        messages = [{"role": "user", "content": "Hello"}]

        params = llm._prepare_response_params(messages)

        assert params["previous_response_id"] == "resp_12345"

    def test_prepare_response_params_with_response_model(self):
        """Test parameter preparation with response model for structured output."""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            answer: str
            confidence: float

        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        params = llm._prepare_response_params(messages, response_model=TestResponse)

        assert "text" in params
        assert params["text"]["format"]["type"] == "json_schema"
        assert params["text"]["format"]["json_schema"]["name"] == "TestResponse"


class TestOpenAIResponsesContextWindow:
    """Tests for context window size."""

    def test_get_context_window_size_gpt4o(self):
        """Test context window size for gpt-4o."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        size = llm.get_context_window_size()
        assert size == int(128000 * 0.85)

    def test_get_context_window_size_o3_mini(self):
        """Test context window size for o3-mini."""
        llm = OpenAIResponsesCompletion(model="o3-mini", api_key="test-key")
        size = llm.get_context_window_size()
        assert size == int(200000 * 0.85)

    def test_get_context_window_size_default(self):
        """Test default context window size for unknown models."""
        llm = OpenAIResponsesCompletion(model="unknown-model", api_key="test-key")
        size = llm.get_context_window_size()
        assert size == int(8192 * 0.85)


class TestOpenAIResponsesFeatureSupport:
    """Tests for feature support methods."""

    def test_supports_function_calling(self):
        """Test that function calling is supported."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        assert llm.supports_function_calling() is True

    def test_supports_stop_words_for_gpt(self):
        """Test that stop words are supported for GPT models."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        assert llm.supports_stop_words() is True

    def test_supports_stop_words_for_o_models(self):
        """Test that stop words are not supported for o-series models."""
        llm = OpenAIResponsesCompletion(model="o3-mini", api_key="test-key")
        assert llm.supports_stop_words() is False


class TestOpenAIResponsesTokenUsage:
    """Tests for token usage extraction."""

    def test_extract_responses_token_usage(self):
        """Test token usage extraction from response."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")

        mock_response = MagicMock()
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=50, total_tokens=150
        )

        usage = llm._extract_responses_token_usage(mock_response)

        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_extract_responses_token_usage_no_usage(self):
        """Test token usage extraction when no usage data."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")

        mock_response = MagicMock()
        mock_response.usage = None

        usage = llm._extract_responses_token_usage(mock_response)

        assert usage["total_tokens"] == 0


class TestOpenAIResponsesMessageFormatting:
    """Tests for message formatting."""

    def test_format_messages_string_input(self):
        """Test formatting of string input."""
        llm = OpenAIResponsesCompletion(model="gpt-4o", api_key="test-key")
        result = llm._format_messages("Hello, world!")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    def test_format_messages_o_model_system_conversion(self):
        """Test that system messages are converted for o-series models."""
        llm = OpenAIResponsesCompletion(model="o3-mini", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        result = llm._format_messages(messages)

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "System: You are helpful."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello!"


class TestOpenAIResponsesClientParams:
    """Tests for client parameter configuration."""

    def test_get_client_params_basic(self):
        """Test basic client parameter configuration."""
        llm = OpenAIResponsesCompletion(
            model="gpt-4o",
            api_key="test-key",
            organization="test-org",
            max_retries=5,
        )

        params = llm._get_client_params()

        assert params["api_key"] == "test-key"
        assert params["organization"] == "test-org"
        assert params["max_retries"] == 5

    def test_get_client_params_with_base_url(self):
        """Test client parameter configuration with base_url."""
        llm = OpenAIResponsesCompletion(
            model="gpt-4o",
            api_key="test-key",
            base_url="https://custom.openai.com/v1",
        )

        params = llm._get_client_params()

        assert params["base_url"] == "https://custom.openai.com/v1"

    def test_get_client_params_api_base_fallback(self):
        """Test that api_base is used as fallback for base_url."""
        llm = OpenAIResponsesCompletion(
            model="gpt-4o",
            api_key="test-key",
            api_base="https://fallback.openai.com/v1",
        )

        params = llm._get_client_params()

        assert params["base_url"] == "https://fallback.openai.com/v1"
