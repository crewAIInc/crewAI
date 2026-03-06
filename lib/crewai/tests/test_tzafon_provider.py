"""Tests for the Tzafon LLM provider integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.llms.base_llm import BaseLLM
from crewai.llms.providers.tzafon.completion import (
    TZAFON_DEFAULT_BASE_URL,
    TzafonCompletion,
)


# ---------------------------------------------------------------------------
# Routing / factory tests
# ---------------------------------------------------------------------------


class TestTzafonRouting:
    """Test that LLM() correctly routes to TzafonCompletion."""

    @patch.dict("os.environ", {"TZAFON_API_KEY": "sk_test_key"})
    def test_prefixed_model_routes_to_tzafon(self):
        """LLM('tzafon/tzafon.sm-1') should create a TzafonCompletion."""
        from crewai.llm import LLM

        llm = LLM("tzafon/tzafon.sm-1")
        assert isinstance(llm, TzafonCompletion)
        assert llm.model == "tzafon.sm-1"

    @patch.dict("os.environ", {"TZAFON_API_KEY": "sk_test_key"})
    def test_explicit_provider_routes_to_tzafon(self):
        """LLM('tzafon.sm-1', provider='tzafon') should create a TzafonCompletion."""
        from crewai.llm import LLM

        llm = LLM("tzafon.sm-1", provider="tzafon")
        assert isinstance(llm, TzafonCompletion)

    @patch.dict("os.environ", {"TZAFON_API_KEY": "sk_test_key"})
    def test_bare_model_inferred_as_tzafon(self):
        """LLM('tzafon.sm-1') without prefix should infer tzafon provider."""
        from crewai.llm import LLM

        llm = LLM("tzafon.sm-1")
        assert isinstance(llm, TzafonCompletion)

    @patch.dict("os.environ", {"TZAFON_API_KEY": "sk_test_key"})
    def test_northstar_model_routes(self):
        """LLM('tzafon/tzafon.northstar-cua-fast') should route correctly."""
        from crewai.llm import LLM

        llm = LLM("tzafon/tzafon.northstar-cua-fast")
        assert isinstance(llm, TzafonCompletion)
        assert llm.model == "tzafon.northstar-cua-fast"

    @patch.dict("os.environ", {"TZAFON_API_KEY": "sk_test_key"})
    def test_is_not_litellm(self):
        """Tzafon provider should be native, not litellm fallback."""
        from crewai.llm import LLM

        llm = LLM("tzafon/tzafon.sm-1")
        assert not getattr(llm, "is_litellm", False)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestTzafonInit:
    """Test TzafonCompletion initialization."""

    def test_api_key_from_param(self):
        """api_key param should be used."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_direct")
        assert llm.api_key == "sk_direct"

    @patch.dict("os.environ", {"TZAFON_API_KEY": "sk_env_key"})
    def test_api_key_from_env(self):
        """TZAFON_API_KEY env var should be picked up."""
        llm = TzafonCompletion(model="tzafon.sm-1")
        assert llm.api_key == "sk_env_key"

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        """Should raise ValueError when no API key is available."""
        with pytest.raises(ValueError, match="Tzafon API key is required"):
            TzafonCompletion(model="tzafon.sm-1")

    def test_default_base_url(self):
        """Default base_url should be the Tzafon API endpoint."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        assert llm.base_url == TZAFON_DEFAULT_BASE_URL

    @patch.dict("os.environ", {"TZAFON_BASE_URL": "https://custom.tzafon.ai/v1"})
    def test_base_url_from_env(self):
        """TZAFON_BASE_URL env var should override default."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        assert llm.base_url == "https://custom.tzafon.ai/v1"

    def test_base_url_from_param(self):
        """base_url param should take priority."""
        llm = TzafonCompletion(
            model="tzafon.sm-1",
            api_key="sk_test",
            base_url="https://override.example.com/v1",
        )
        assert llm.base_url == "https://override.example.com/v1"

    def test_extends_base_llm(self):
        """TzafonCompletion should be a subclass of BaseLLM."""
        assert issubclass(TzafonCompletion, BaseLLM)

    def test_temperature_param(self):
        """Temperature parameter should be stored."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", temperature=0.5
        )
        assert llm.temperature == 0.5

    def test_max_tokens_param(self):
        """max_tokens parameter should be stored."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", max_tokens=1024
        )
        assert llm.max_tokens == 1024

    def test_stream_param(self):
        """stream parameter should be stored."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", stream=True
        )
        assert llm.stream is True


# ---------------------------------------------------------------------------
# Context window / capabilities tests
# ---------------------------------------------------------------------------


class TestTzafonCapabilities:
    """Test capability reporting methods."""

    def test_context_window_size(self):
        """get_context_window_size should return a positive integer."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        size = llm.get_context_window_size()
        assert isinstance(size, int)
        assert size > 0

    def test_supports_stop_words_without_stop(self):
        """supports_stop_words should be False when no stop words configured."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        assert llm.supports_stop_words() is False

    def test_supports_stop_words_with_stop(self):
        """supports_stop_words should be True when stop words are configured."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", stop=["Observation:"]
        )
        assert llm.supports_stop_words() is True

    def test_supports_multimodal(self):
        """Tzafon should report no multimodal support."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        assert llm.supports_multimodal() is False


# ---------------------------------------------------------------------------
# Call tests (mocked)
# ---------------------------------------------------------------------------


def _make_mock_response(content="Hello from Tzafon", usage=None):
    """Create a mock ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    if usage is None:
        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 10
        usage_obj.completion_tokens = 20
        usage_obj.total_tokens = 30
    else:
        usage_obj = MagicMock(**usage)

    response.usage = usage_obj
    return response


class TestTzafonCall:
    """Test the call() method with mocked OpenAI client."""

    def test_basic_call(self):
        """Basic call should return response content."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        mock_response = _make_mock_response("Hello!")
        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        result = llm.call("Say hello")
        assert "Hello!" in result
        llm.client.chat.completions.create.assert_called_once()

    def test_call_with_message_list(self):
        """Should handle list of message dicts."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        mock_response = _make_mock_response("Response")
        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = llm.call(messages)
        assert result == "Response"

    def test_token_usage_tracked(self):
        """Token usage should be tracked after a call."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        mock_response = _make_mock_response("test")
        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        llm.call("test")

        usage = llm.get_token_usage_summary()
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        assert usage.successful_requests == 1

    def test_stop_words_applied(self):
        """Stop words should truncate the response."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", stop=["Observation:"]
        )
        mock_response = _make_mock_response(
            "I need to search.\n\nObservation: Found results"
        )
        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        result = llm.call("test")
        assert "Observation:" not in result
        assert "I need to search." in result

    def test_call_passes_temperature(self):
        """Temperature should be passed in API params."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", temperature=0.3
        )
        mock_response = _make_mock_response("ok")
        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        llm.call("test")

        call_kwargs = llm.client.chat.completions.create.call_args
        assert call_kwargs[1]["temperature"] == 0.3

    def test_call_passes_max_tokens(self):
        """max_tokens should be passed in API params."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", max_tokens=512
        )
        mock_response = _make_mock_response("ok")
        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        llm.call("test")

        call_kwargs = llm.client.chat.completions.create.call_args
        assert call_kwargs[1]["max_tokens"] == 512

    def test_call_passes_model(self):
        """Model name should be passed in API params."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        mock_response = _make_mock_response("ok")
        llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        llm.call("test")

        call_kwargs = llm.client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "tzafon.sm-1"


# ---------------------------------------------------------------------------
# Streaming tests (mocked)
# ---------------------------------------------------------------------------


class TestTzafonStreaming:
    """Test streaming functionality."""

    def test_streaming_call(self):
        """Streaming should accumulate chunks into a full response."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", stream=True
        )

        # Create mock chunks
        chunks = []
        for text in ["Hello", " from", " Tzafon"]:
            chunk = MagicMock()
            chunk.id = "chatcmpl-123"
            chunk.usage = None
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunk.choices[0].delta.tool_calls = None
            chunks.append(chunk)

        # Add final usage chunk
        usage_chunk = MagicMock()
        usage_chunk.id = "chatcmpl-123"
        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 5
        usage_obj.completion_tokens = 3
        usage_obj.total_tokens = 8
        usage_chunk.usage = usage_obj
        usage_chunk.choices = []
        chunks.append(usage_chunk)

        llm.client.chat.completions.create = MagicMock(return_value=iter(chunks))

        result = llm.call("test")
        assert result == "Hello from Tzafon"

        usage = llm.get_token_usage_summary()
        assert usage.total_tokens == 8


# ---------------------------------------------------------------------------
# Tool calling tests (mocked)
# ---------------------------------------------------------------------------


class TestTzafonToolCalling:
    """Test tool calling functionality."""

    def test_tool_call_execution(self):
        """Tool calls should be executed when available_functions provided."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")

        # Create mock response with tool call
        tool_call = MagicMock()
        tool_call.function.name = "search"
        tool_call.function.arguments = json.dumps({"query": "test"})

        message = MagicMock()
        message.content = None
        message.tool_calls = [tool_call]

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]
        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 10
        usage_obj.completion_tokens = 5
        usage_obj.total_tokens = 15
        response.usage = usage_obj

        llm.client.chat.completions.create = MagicMock(return_value=response)

        def mock_search(query: str) -> str:
            return f"Results for: {query}"

        result = llm.call(
            "Search for test",
            available_functions={"search": mock_search},
        )

        assert "Results for: test" in result

    def test_tool_calls_returned_without_functions(self):
        """Tool calls should be returned raw when no available_functions."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")

        tool_call = MagicMock()
        tool_call.function.name = "search"
        tool_call.function.arguments = json.dumps({"query": "test"})

        message = MagicMock()
        message.content = None
        message.tool_calls = [tool_call]

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]
        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 10
        usage_obj.completion_tokens = 5
        usage_obj.total_tokens = 15
        response.usage = usage_obj

        llm.client.chat.completions.create = MagicMock(return_value=response)

        result = llm.call("Search for test")
        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Prepare params tests
# ---------------------------------------------------------------------------


class TestPrepareParams:
    """Test _prepare_params method."""

    def test_basic_params(self):
        """Should include model and messages."""
        llm = TzafonCompletion(model="tzafon.sm-1", api_key="sk_test")
        messages = [{"role": "user", "content": "hello"}]
        params = llm._prepare_params(messages=messages)

        assert params["model"] == "tzafon.sm-1"
        assert params["messages"] == messages

    def test_stream_params(self):
        """Should include stream options when streaming."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", stream=True
        )
        messages = [{"role": "user", "content": "hello"}]
        params = llm._prepare_params(messages=messages)

        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": True}

    def test_stop_in_params(self):
        """Stop sequences should be included in params."""
        llm = TzafonCompletion(
            model="tzafon.sm-1", api_key="sk_test", stop=["END"]
        )
        messages = [{"role": "user", "content": "hello"}]
        params = llm._prepare_params(messages=messages)

        assert params["stop"] == ["END"]
