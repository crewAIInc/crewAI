"""Tests for remote Ollama server support (Issue #4694).

Bug 1: InternalInstructor.to_pydantic() doesn't forward api_base/base_url to litellm.
Bug 2: LLM.supports_function_calling() doesn't query remote Ollama for capabilities.
"""

from unittest.mock import Mock, patch

import httpx
from pydantic import BaseModel

from crewai.llm import LLM
from crewai.utilities.internal_instructor import InternalInstructor


class SimpleModel(BaseModel):
    name: str
    age: int


# =====================================================================
# Bug 1: InternalInstructor forwards api_base/base_url to litellm
# =====================================================================


def _make_instructor_bypass_init(llm: object) -> "InternalInstructor[SimpleModel]":
    """Create an InternalInstructor bypassing __init__ to avoid instructor import.

    This is useful for testing helper methods without needing to mock instructor.
    """
    inst: InternalInstructor[SimpleModel] = object.__new__(InternalInstructor)
    inst.content = "test"
    inst.model = SimpleModel
    inst.llm = llm
    inst.agent = None
    inst._client = Mock()
    return inst


class TestInternalInstructorForwardsApiBase:
    """Test that InternalInstructor passes api_base/base_url to litellm completion."""

    def _make_litellm_instructor(
        self, mock_llm: Mock, mock_client: Mock
    ) -> "InternalInstructor[SimpleModel]":
        """Create an InternalInstructor with litellm path, mock client injected."""
        inst = _make_instructor_bypass_init(mock_llm)
        inst._client = mock_client
        return inst

    def test_to_pydantic_forwards_api_base(self) -> None:
        """When LLM has api_base set, it should be forwarded to the create() call."""
        mock_llm = Mock()
        mock_llm.is_litellm = True
        mock_llm.model = "ollama_chat/mistral-small3.2:24b"
        mock_llm.api_base = "http://remote-server:11434"
        mock_llm.base_url = None
        mock_llm.api_key = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SimpleModel(
            name="Test", age=25
        )

        inst = self._make_litellm_instructor(mock_llm, mock_client)
        result = inst.to_pydantic()

        assert isinstance(result, SimpleModel)
        assert result.name == "Test"
        assert result.age == 25

        # Verify api_base was forwarded
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("api_base") == "http://remote-server:11434"

    def test_to_pydantic_forwards_base_url(self) -> None:
        """When LLM has base_url set, it should be forwarded to the create() call."""
        mock_llm = Mock()
        mock_llm.is_litellm = True
        mock_llm.model = "ollama/mistral-small3.2:24b"
        mock_llm.api_base = None
        mock_llm.base_url = "http://remote-server:11434"
        mock_llm.api_key = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SimpleModel(
            name="Test", age=30
        )

        inst = self._make_litellm_instructor(mock_llm, mock_client)
        result = inst.to_pydantic()

        assert isinstance(result, SimpleModel)
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("base_url") == "http://remote-server:11434"

    def test_to_pydantic_forwards_api_key(self) -> None:
        """When LLM has api_key set, it should be forwarded to the create() call."""
        mock_llm = Mock()
        mock_llm.is_litellm = True
        mock_llm.model = "ollama/mistral-small3.2:24b"
        mock_llm.api_base = "http://remote-server:11434"
        mock_llm.base_url = None
        mock_llm.api_key = "test-key-123"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SimpleModel(
            name="Test", age=35
        )

        inst = self._make_litellm_instructor(mock_llm, mock_client)
        inst.to_pydantic()

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("api_base") == "http://remote-server:11434"
        assert call_kwargs.kwargs.get("api_key") == "test-key-123"

    def test_to_pydantic_no_extra_kwargs_for_string_llm(self) -> None:
        """When LLM is a string, no extra kwargs should be passed."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SimpleModel(
            name="Test", age=20
        )

        inst = _make_instructor_bypass_init("openai/gpt-4o")
        inst._client = mock_client
        inst.to_pydantic()

        call_kwargs = mock_client.chat.completions.create.call_args
        # No extra kwargs should be present for string LLM
        assert "api_base" not in (call_kwargs.kwargs or {})
        assert "base_url" not in (call_kwargs.kwargs or {})
        assert "api_key" not in (call_kwargs.kwargs or {})

    def test_to_pydantic_no_extra_kwargs_when_none(self) -> None:
        """When LLM has no api_base/base_url/api_key, no extra kwargs should be passed."""
        mock_llm = Mock()
        mock_llm.is_litellm = True
        mock_llm.model = "gpt-4o"
        mock_llm.api_base = None
        mock_llm.base_url = None
        mock_llm.api_key = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SimpleModel(
            name="Test", age=40
        )

        inst = self._make_litellm_instructor(mock_llm, mock_client)
        inst.to_pydantic()

        call_kwargs = mock_client.chat.completions.create.call_args
        assert "api_base" not in (call_kwargs.kwargs or {})
        assert "base_url" not in (call_kwargs.kwargs or {})
        assert "api_key" not in (call_kwargs.kwargs or {})


class TestGetLlmExtraKwargs:
    """Test the _get_llm_extra_kwargs helper method."""

    def test_returns_empty_for_none_llm(self) -> None:
        inst = _make_instructor_bypass_init(llm=None)
        assert inst._get_llm_extra_kwargs() == {}

    def test_returns_empty_for_string_llm(self) -> None:
        inst = _make_instructor_bypass_init(llm="gpt-4o")
        assert inst._get_llm_extra_kwargs() == {}

    def test_returns_empty_for_non_litellm(self) -> None:
        mock_llm = Mock()
        mock_llm.is_litellm = False
        mock_llm.api_base = "http://remote:11434"
        mock_llm.base_url = None
        mock_llm.api_key = None

        inst = _make_instructor_bypass_init(llm=mock_llm)
        assert inst._get_llm_extra_kwargs() == {}

    def test_returns_api_base_when_set(self) -> None:
        mock_llm = Mock()
        mock_llm.is_litellm = True
        mock_llm.api_base = "http://remote:11434"
        mock_llm.base_url = None
        mock_llm.api_key = None

        inst = _make_instructor_bypass_init(llm=mock_llm)
        extra = inst._get_llm_extra_kwargs()
        assert extra == {"api_base": "http://remote:11434"}

    def test_returns_multiple_attrs_when_set(self) -> None:
        mock_llm = Mock()
        mock_llm.is_litellm = True
        mock_llm.api_base = "http://remote:11434"
        mock_llm.base_url = "http://remote:11434"
        mock_llm.api_key = "secret"

        inst = _make_instructor_bypass_init(llm=mock_llm)
        extra = inst._get_llm_extra_kwargs()
        assert extra == {
            "api_base": "http://remote:11434",
            "base_url": "http://remote:11434",
            "api_key": "secret",
        }


# =====================================================================
# Bug 2: LLM.supports_function_calling() with remote Ollama
# =====================================================================


class TestIsOllamaModel:
    """Test the _is_ollama_model helper method."""

    def test_ollama_prefix(self) -> None:
        llm = LLM(model="ollama/mistral-small3.2:24b", is_litellm=True)
        assert llm._is_ollama_model() is True

    def test_ollama_chat_prefix(self) -> None:
        llm = LLM(model="ollama_chat/mistral-small3.2:24b", is_litellm=True)
        assert llm._is_ollama_model() is True

    def test_non_ollama_model(self) -> None:
        llm = LLM(model="gpt-4o", is_litellm=True)
        assert llm._is_ollama_model() is False

    def test_openai_model(self) -> None:
        llm = LLM(model="openai/gpt-4o", is_litellm=True)
        assert llm._is_ollama_model() is False


class TestGetOllamaBaseUrl:
    """Test the _get_ollama_base_url helper method."""

    def test_returns_api_base(self) -> None:
        llm = LLM(
            model="ollama/mistral",
            api_base="http://remote:11434",
            is_litellm=True,
        )
        assert llm._get_ollama_base_url() == "http://remote:11434"

    def test_returns_base_url(self) -> None:
        llm = LLM(
            model="ollama/mistral",
            base_url="http://remote:11434",
            is_litellm=True,
        )
        assert llm._get_ollama_base_url() == "http://remote:11434"

    def test_api_base_takes_precedence(self) -> None:
        llm = LLM(
            model="ollama/mistral",
            api_base="http://api-base:11434",
            base_url="http://base-url:11434",
            is_litellm=True,
        )
        assert llm._get_ollama_base_url() == "http://api-base:11434"

    def test_returns_none_when_not_set(self) -> None:
        llm = LLM(model="ollama/mistral", is_litellm=True)
        assert llm._get_ollama_base_url() is None


class TestCheckOllamaFunctionCalling:
    """Test the _check_ollama_function_calling method."""

    def test_returns_true_when_tool_key_in_model_info(self) -> None:
        """Remote Ollama returns model_info with tool-related capability."""
        llm = LLM(
            model="ollama/mistral-small3.2:24b",
            api_base="http://remote:11434",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_info": {
                "general.architecture": "mistral",
                "mistral.tool_call": True,
            },
            "template": "",
        }

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = llm._check_ollama_function_calling()
            assert result is True
            mock_post.assert_called_once_with(
                "http://remote:11434/api/show",
                json={"name": "mistral-small3.2:24b"},
                timeout=5.0,
            )

    def test_returns_true_when_tools_in_template(self) -> None:
        """Remote Ollama returns template mentioning tools."""
        llm = LLM(
            model="ollama_chat/mistral-small3.2:24b",
            api_base="http://remote:11434",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_info": {},
            "template": "{{- if .ToolCalls }}\n[TOOL_CALLS]{{ range .ToolCalls }}",
        }

        with patch("httpx.post", return_value=mock_response):
            assert llm._check_ollama_function_calling() is True

    def test_returns_true_when_tools_keyword_in_template(self) -> None:
        """Remote Ollama returns template with 'tools' keyword."""
        llm = LLM(
            model="ollama/qwen2.5:32b",
            api_base="http://remote:11434",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_info": {},
            "template": "{{- if .Tools }}\nAvailable tools:\n{{ range .Tools }}",
        }

        with patch("httpx.post", return_value=mock_response):
            assert llm._check_ollama_function_calling() is True

    def test_returns_false_when_no_tool_support(self) -> None:
        """Remote Ollama model without tool support."""
        llm = LLM(
            model="ollama/llama2",
            api_base="http://remote:11434",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_info": {"general.architecture": "llama"},
            "template": "{{ .Prompt }}",
        }

        with patch("httpx.post", return_value=mock_response):
            assert llm._check_ollama_function_calling() is False

    def test_returns_false_when_no_base_url(self) -> None:
        """When no base URL is configured, should return False."""
        llm = LLM(model="ollama/mistral", is_litellm=True)
        assert llm._check_ollama_function_calling() is False

    def test_returns_false_on_connection_error(self) -> None:
        """When the remote server is unreachable, should return False."""
        llm = LLM(
            model="ollama/mistral",
            api_base="http://unreachable:11434",
            is_litellm=True,
        )

        with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
            assert llm._check_ollama_function_calling() is False

    def test_returns_false_on_404(self) -> None:
        """When the remote server returns 404, should return False."""
        llm = LLM(
            model="ollama/mistral",
            api_base="http://remote:11434",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 404

        with patch("httpx.post", return_value=mock_response):
            assert llm._check_ollama_function_calling() is False

    def test_strips_trailing_slash_from_url(self) -> None:
        """Trailing slash in api_base should be stripped."""
        llm = LLM(
            model="ollama/mistral",
            api_base="http://remote:11434/",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_info": {"mistral.tool_call": True},
            "template": "",
        }

        with patch("httpx.post", return_value=mock_response) as mock_post:
            llm._check_ollama_function_calling()
            mock_post.assert_called_once_with(
                "http://remote:11434/api/show",
                json={"name": "mistral"},
                timeout=5.0,
            )


class TestSupportsFunctionCallingWithRemoteOllama:
    """Test the full supports_function_calling flow with remote Ollama."""

    def test_litellm_returns_true_no_fallback_needed(self) -> None:
        """When litellm says the model supports function calling, use that."""
        llm = LLM(model="gpt-4o", is_litellm=True)
        with patch(
            "litellm.utils.supports_function_calling", return_value=True
        ):
            assert llm.supports_function_calling() is True

    def test_remote_ollama_fallback_when_litellm_returns_false(self) -> None:
        """When litellm returns False for Ollama with remote URL, query the server."""
        llm = LLM(
            model="ollama_chat/mistral-small3.2:24b",
            api_base="http://remote:11434",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_info": {"mistral.tool_call": True},
            "template": "",
        }

        with patch("litellm.utils.supports_function_calling", return_value=False):
            with patch("httpx.post", return_value=mock_response):
                assert llm.supports_function_calling() is True

    def test_no_fallback_for_non_ollama_models(self) -> None:
        """When litellm returns False for non-Ollama models, don't use fallback."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)

        with patch("litellm.utils.supports_function_calling", return_value=False):
            assert llm.supports_function_calling() is False

    def test_no_fallback_without_remote_url(self) -> None:
        """Ollama model without remote URL shouldn't trigger fallback."""
        llm = LLM(model="ollama/mistral", is_litellm=True)

        with patch("litellm.utils.supports_function_calling", return_value=False):
            assert llm.supports_function_calling() is False

    def test_fallback_returns_false_when_model_lacks_tools(self) -> None:
        """Ollama model that doesn't support tools returns False even with fallback."""
        llm = LLM(
            model="ollama/llama2",
            api_base="http://remote:11434",
            is_litellm=True,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_info": {"general.architecture": "llama"},
            "template": "{{ .Prompt }}",
        }

        with patch("litellm.utils.supports_function_calling", return_value=False):
            with patch("httpx.post", return_value=mock_response):
                assert llm.supports_function_calling() is False

    def test_exception_handling(self) -> None:
        """When litellm raises an exception, should return False gracefully."""
        llm = LLM(model="ollama/mistral", api_base="http://remote:11434", is_litellm=True)

        with patch(
            "litellm.utils.supports_function_calling",
            side_effect=Exception("Unexpected error"),
        ):
            assert llm.supports_function_calling() is False
