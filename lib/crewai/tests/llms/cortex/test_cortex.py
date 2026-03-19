import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from crewai.llm import LLM
from crewai.llms.constants import CORTEX_MODELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_snowflake_credentials():
    """Set mock Snowflake credentials for all tests."""
    env = {
        "SNOWFLAKE_ACCOUNT": "testaccount",
        "SNOWFLAKE_USER": "testuser",
        "SNOWFLAKE_PAT": "test-pat-token",
    }
    with patch.dict(os.environ, env):
        yield


def _mock_response(body: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", "https://testaccount.snowflakecomputing.com/api/v2/cortex/inference:complete"),
    )


def _text_response(text: str = "Hello from Cortex!") -> dict:
    """Standard text completion response."""
    return {
        "choices": [{"messages": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_call_response(name: str = "get_weather", args: dict | None = None) -> dict:
    """Standard tool call response."""
    return {
        "choices": [{
            "messages": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args or {"location": "SF"}),
                    },
                }],
            },
            "finish_reason": "tool_use",
        }],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }


# ---------------------------------------------------------------------------
# Factory routing tests
# ---------------------------------------------------------------------------

class TestCortexFactoryRouting:
    """Test that the LLM factory correctly routes to CortexCompletion."""

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_explicit_provider_kwarg(self, mock_client_cls):
        """LLM(model='llama3.1-70b', provider='cortex') uses CortexCompletion."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = LLM(model="llama3.1-70b", provider="cortex")
        assert isinstance(llm, CortexCompletion)

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_cortex_prefix_routing(self, mock_client_cls):
        """LLM(model='cortex/claude-3-5-sonnet') routes to CortexCompletion."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = LLM(model="cortex/claude-3-5-sonnet")
        assert isinstance(llm, CortexCompletion)
        assert llm.model == "claude-3-5-sonnet"

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_snowflake_prefix_routing(self, mock_client_cls):
        """LLM(model='snowflake/llama3.1-70b') routes to CortexCompletion."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = LLM(model="snowflake/llama3.1-70b")
        assert isinstance(llm, CortexCompletion)
        assert llm.model == "llama3.1-70b"

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_snowflake_provider_kwarg(self, mock_client_cls):
        """provider='snowflake' maps to cortex."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = LLM(model="mistral-large2", provider="snowflake")
        assert isinstance(llm, CortexCompletion)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestCortexConstants:
    """Test Cortex model constants."""

    def test_cortex_models_not_empty(self):
        assert len(CORTEX_MODELS) > 0

    def test_known_models_in_constants(self):
        expected = [
            "claude-3-5-sonnet", "claude-3-7-sonnet", "llama3.1-70b",
            "mistral-large2", "snowflake-arctic",
        ]
        for model in expected:
            assert model in CORTEX_MODELS, f"{model} missing from CORTEX_MODELS"

    def test_infer_provider_from_cortex_model(self):
        assert LLM._infer_provider_from_model("snowflake-arctic") == "cortex"
        assert LLM._infer_provider_from_model("llama3.1-70b") == "cortex"
        assert LLM._infer_provider_from_model("claude-3-5-sonnet") == "cortex"

    def test_validate_model_in_constants(self):
        assert LLM._validate_model_in_constants("claude-3-5-sonnet", "cortex") is True
        assert LLM._validate_model_in_constants("llama3.1-70b", "cortex") is True

    def test_matches_provider_pattern(self):
        assert LLM._matches_provider_pattern("llama3.1-8b", "cortex") is True
        assert LLM._matches_provider_pattern("snowflake-arctic", "cortex") is True
        assert LLM._matches_provider_pattern("reka-flash", "cortex") is True
        assert LLM._matches_provider_pattern("gpt-4o", "cortex") is False


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestCortexCompletionInit:
    """Test CortexCompletion initialization and auth."""

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_default_init(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(model="claude-3-5-sonnet")
        assert llm.model == "claude-3-5-sonnet"
        assert llm.account == "testaccount"
        assert llm.max_tokens == 4096

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_custom_params(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(
            model="llama3.1-70b",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
        )
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2048
        assert llm.top_p == 0.9

    def test_missing_account_raises(self):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Snowflake account is required"):
                CortexCompletion(model="claude-3-5-sonnet")

    def test_missing_auth_raises(self):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        with patch.dict(os.environ, {"SNOWFLAKE_ACCOUNT": "testaccount"}, clear=True):
            with pytest.raises(ValueError, match="No Snowflake authentication configured"):
                CortexCompletion(model="claude-3-5-sonnet")

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_pat_auth_token_type(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(model="claude-3-5-sonnet")
        assert llm._token_type == "PROGRAMMATIC_ACCESS_TOKEN"

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_api_key_alias(self, mock_client_cls):
        """api_key kwarg should work as alias for pat."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        with patch.dict(os.environ, {"SNOWFLAKE_ACCOUNT": "testaccount"}, clear=True):
            llm = CortexCompletion(model="claude-3-5-sonnet", api_key="my-pat-token")
            assert llm.pat == "my-pat-token"

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_custom_base_url(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(
            model="claude-3-5-sonnet",
            base_url="https://custom.snowflake.example.com/",
        )
        assert llm.base_url == "https://custom.snowflake.example.com"


# ---------------------------------------------------------------------------
# API call tests
# ---------------------------------------------------------------------------

class TestCortexCompletionCall:
    """Test CortexCompletion.call() with mocked HTTP responses."""

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_text_completion(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(_text_response("Hello!"))
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="llama3.1-70b")
        result = llm.call("Say hello")

        assert result == "Hello!"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/api/v2/cortex/inference:complete"

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_request_payload_structure(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(_text_response())
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="claude-3-5-sonnet", temperature=0.7, max_tokens=1024)
        llm.call("Test")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "claude-3-5-sonnet"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 1024
        assert "messages" in payload

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_dict_response_with_content(self, mock_client_cls):
        """Test handling a dict response (non-tool-call)."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        response_body = {
            "choices": [{
                "messages": {"role": "assistant", "content": "Dict response"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(response_body)
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="claude-3-5-sonnet")
        result = llm.call("Test")

        assert result == "Dict response"

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_tool_call_execution(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            _tool_call_response("get_weather", {"location": "San Francisco"})
        )
        mock_client_cls.return_value = mock_client

        def get_weather(location: str) -> str:
            return f"Sunny in {location}"

        llm = CortexCompletion(model="claude-3-5-sonnet")
        result = llm.call(
            "What is the weather in SF?",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }],
            available_functions={"get_weather": get_weather},
        )

        assert result == "Sunny in San Francisco"

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_stop_words_applied(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            _text_response("I found the answer.\nObservation: more text")
        )
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="llama3.1-70b", stop=["Observation:"])
        result = llm.call("Test")

        assert "Observation:" not in result
        assert result == "I found the answer."

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_empty_choices_raises(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response({"choices": [], "usage": {}})
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="llama3.1-70b")
        with pytest.raises(RuntimeError, match="empty choices"):
            llm.call("Test")

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_token_usage_tracking(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(_text_response())
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="llama3.1-70b")
        llm.call("Test")

        assert llm._token_usage["prompt_tokens"] == 10
        assert llm._token_usage["completion_tokens"] == 5
        assert llm._token_usage["total_tokens"] == 15


# ---------------------------------------------------------------------------
# Tool conversion tests
# ---------------------------------------------------------------------------

class TestCortexToolConversion:
    """Test tool format conversion to Cortex tool_spec format."""

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_tools_not_sent_for_non_tool_model(self, mock_client_cls):
        """Non-tool-calling models should not get tools in payload."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(_text_response())
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="llama3.1-70b")
        llm.call(
            "Test",
            tools=[{
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "A tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        )

        payload = mock_client.post.call_args[1]["json"]
        assert "tools" not in payload

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_tool_spec_passthrough(self, mock_client_cls):
        """Tools already in Cortex format should pass through."""
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(_text_response())
        mock_client_cls.return_value = mock_client

        cortex_tool = {
            "tool_spec": {
                "type": "function",
                "function": {"name": "my_tool", "description": "A tool", "parameters": {}},
            }
        }

        llm = CortexCompletion(model="claude-3-5-sonnet")
        llm.call("Test", tools=[cortex_tool])

        payload = mock_client.post.call_args[1]["json"]
        assert payload["tools"][0] == cortex_tool


# ---------------------------------------------------------------------------
# Retry / error handling tests
# ---------------------------------------------------------------------------

class TestCortexRetry:
    """Test retry and error handling logic."""

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_401_triggers_token_refresh(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        # First call returns 401, second succeeds
        mock_client.post.side_effect = [
            _mock_response({"error": "unauthorized"}, status_code=401),
            _mock_response(_text_response("Recovered")),
        ]
        mock_client.headers = {}
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="llama3.1-70b")
        result = llm.call("Test")

        assert result == "Recovered"
        assert mock_client.post.call_count == 2

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_http_error_raises_runtime_error(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        mock_client = MagicMock()
        error_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "https://test.snowflakecomputing.com/api/v2/cortex/inference:complete"),
        )
        mock_client.post.return_value = error_response
        mock_client_cls.return_value = mock_client

        llm = CortexCompletion(model="llama3.1-70b")
        with pytest.raises(RuntimeError, match="Cortex API"):
            llm.call("Test")


# ---------------------------------------------------------------------------
# Provider capability tests
# ---------------------------------------------------------------------------

class TestCortexCapabilities:
    """Test model-specific capabilities."""

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_tool_calling_supported_models(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(model="claude-3-5-sonnet")
        assert llm._supports_tool_calling() is True

        llm2 = CortexCompletion(model="claude-3-7-sonnet")
        assert llm2._supports_tool_calling() is True

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_tool_calling_unsupported_models(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(model="llama3.1-70b")
        assert llm._supports_tool_calling() is False

        llm2 = CortexCompletion(model="snowflake-arctic")
        assert llm2._supports_tool_calling() is False

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_context_window_known_model(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(model="claude-3-5-sonnet")
        assert llm.get_context_window_size() == 200_000

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_context_window_unknown_model(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(model="some-future-model")
        assert llm.get_context_window_size() == 128_000  # default fallback

    @patch("crewai.llms.providers.cortex.completion.httpx.Client")
    def test_supports_stop_words(self, mock_client_cls):
        from crewai.llms.providers.cortex.completion import CortexCompletion

        llm = CortexCompletion(model="llama3.1-70b")
        assert llm.supports_stop_words() is False

        llm2 = CortexCompletion(model="llama3.1-70b", stop=["END"])
        assert llm2.supports_stop_words() is True
