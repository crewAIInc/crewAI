"""Tests for Azure OpenAI Responses API support.

Verifies that AzureCompletion with api='responses' correctly delegates
to OpenAICompletion configured with the Azure OpenAI /openai/v1/ base URL.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def azure_env():
    """Set Azure environment variables for tests."""
    with patch.dict(
        os.environ,
        {
            "AZURE_API_KEY": "test-azure-key",
            "AZURE_ENDPOINT": "https://myresource.openai.azure.com",
        },
    ):
        yield


@pytest.fixture
def mock_openai_completion():
    """Mock OpenAICompletion to avoid real client creation.

    Patches at the source module so that the dynamic import inside
    _init_responses_delegate picks up the mock.
    """
    instance = MagicMock()
    instance.call = MagicMock(return_value="responses-result")
    instance.acall = AsyncMock(return_value="async-responses-result")
    instance.last_response_id = "resp_abc123"
    instance.last_reasoning_items = [{"type": "reasoning"}]
    instance.reset_chain = MagicMock()
    instance.reset_reasoning_chain = MagicMock()
    mock_cls = MagicMock(return_value=instance)

    with patch(
        "crewai.llms.providers.openai.completion.OpenAICompletion",
        mock_cls,
    ):
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# Helper to build AzureCompletion with api="responses" while mocking imports
# ---------------------------------------------------------------------------


def _create_azure_responses(**overrides):
    """Create an AzureCompletion(api='responses').

    Must be called inside a context where OpenAICompletion is already mocked
    (i.e. via the ``mock_openai_completion`` fixture).
    """
    from crewai.llms.providers.azure.completion import AzureCompletion

    defaults = {
        "model": "gpt-4o",
        "api_key": "test-azure-key",
        "endpoint": "https://myresource.openai.azure.com",
        "api": "responses",
    }
    defaults.update(overrides)
    return AzureCompletion(**defaults)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestAzureResponsesInit:
    """Test initialization with api='responses'."""

    def test_default_api_is_completions(self):
        """Default api should be 'completions' (existing behaviour)."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        comp = AzureCompletion(
            model="gpt-4o",
            api_key="key",
            endpoint="https://res.openai.azure.com",
        )
        assert comp.api == "completions"
        assert comp._responses_delegate is None

    def test_responses_api_creates_delegate(self, mock_openai_completion):
        mock_cls, instance = mock_openai_completion
        comp = _create_azure_responses()

        assert comp.api == "responses"
        assert comp._responses_delegate is instance
        mock_cls.assert_called_once()

    def test_completions_clients_not_created_in_responses_mode(
        self, mock_openai_completion
    ):
        """When api='responses', azure-ai-inference clients should not be created."""
        _mock_cls, _ = mock_openai_completion
        comp = _create_azure_responses()

        assert comp._client is None
        assert comp._async_client is None

    def test_responses_base_url_from_base_endpoint(self, mock_openai_completion):
        mock_cls, _ = mock_openai_completion
        _create_azure_responses(
            endpoint="https://myresource.openai.azure.com",
        )
        call_kwargs = mock_cls.call_args[1]
        assert (
            call_kwargs["base_url"] == "https://myresource.openai.azure.com/openai/v1/"
        )

    def test_responses_base_url_strips_deployment_path(self, mock_openai_completion):
        """Endpoint with /openai/deployments/... should still produce correct base_url."""
        mock_cls, _ = mock_openai_completion
        _create_azure_responses(
            endpoint="https://myresource.openai.azure.com/openai/deployments/gpt-4o",
        )
        call_kwargs = mock_cls.call_args[1]
        assert (
            call_kwargs["base_url"] == "https://myresource.openai.azure.com/openai/v1/"
        )

    def test_responses_base_url_preserves_port(self, mock_openai_completion):
        mock_cls, _ = mock_openai_completion
        _create_azure_responses(
            endpoint="https://myresource.openai.azure.com:8443/openai/deployments/gpt-4o",
        )
        call_kwargs = mock_cls.call_args[1]
        assert (
            call_kwargs["base_url"]
            == "https://myresource.openai.azure.com:8443/openai/v1/"
        )

    def test_delegate_receives_model_and_api_key(self, mock_openai_completion):
        mock_cls, _ = mock_openai_completion
        _create_azure_responses(
            model="gpt-4o",
            api_key="my-key",
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["api_key"] == "my-key"
        assert call_kwargs["api"] == "responses"
        assert call_kwargs["provider"] == "openai"

    def test_delegate_receives_optional_params(self, mock_openai_completion):
        mock_cls, _ = mock_openai_completion
        _create_azure_responses(
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            max_completion_tokens=800,
            reasoning_effort="medium",
            instructions="Be helpful",
            store=True,
            previous_response_id="resp_prev",
            include=["reasoning.encrypted_content"],
            builtin_tools=["web_search"],
            parse_tool_outputs=True,
            auto_chain=True,
            auto_chain_reasoning=True,
            stream=True,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["max_completion_tokens"] == 800
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["instructions"] == "Be helpful"
        assert call_kwargs["store"] is True
        assert call_kwargs["previous_response_id"] == "resp_prev"
        assert call_kwargs["include"] == ["reasoning.encrypted_content"]
        assert call_kwargs["builtin_tools"] == ["web_search"]
        assert call_kwargs["parse_tool_outputs"] is True
        assert call_kwargs["auto_chain"] is True
        assert call_kwargs["auto_chain_reasoning"] is True
        assert call_kwargs["stream"] is True

    def test_delegate_omits_unset_optional_params(self, mock_openai_completion):
        """Params left at defaults should not be passed to the delegate."""
        mock_cls, _ = mock_openai_completion
        _create_azure_responses()
        call_kwargs = mock_cls.call_args[1]
        # These should NOT be in kwargs because they were not set
        assert "temperature" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs
        assert "instructions" not in call_kwargs
        assert "store" not in call_kwargs
        assert "max_completion_tokens" not in call_kwargs


# ---------------------------------------------------------------------------
# Call delegation tests (VCR cassette-based)
# ---------------------------------------------------------------------------


class TestAzureResponsesCall:
    """Test call / acall delegation to the Responses API using VCR cassettes."""

    @pytest.mark.vcr()
    def test_call_delegates_to_responses(self):
        from crewai.llm import LLM

        llm = LLM(model="azure/gpt-5.2-chat", api="responses")
        result = llm.call("Say hello in one sentence.")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_call_with_tools_delegates(self):
        from crewai.llm import LLM

        llm = LLM(
            model="azure/gpt-5.2-chat",
            api="responses",
            builtin_tools=["web_search"],
        )
        result = llm.call("What is 2 + 2? Be brief.")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_completions_call_unchanged(self):
        """Default api='completions' should not use the responses delegate."""
        from crewai.llm import LLM

        llm = LLM(model="azure/gpt-5.2-chat")
        result = llm.call("Say hello in one sentence.")

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Delegated property & method tests
# ---------------------------------------------------------------------------


class TestAzureResponsesProperties:
    """Test properties and methods delegated to the responses delegate."""

    def test_last_response_id(self, mock_openai_completion):
        _mock_cls, _ = mock_openai_completion
        comp = _create_azure_responses()
        assert comp.last_response_id == "resp_abc123"

    def test_last_response_id_none_for_completions(self):
        from crewai.llms.providers.azure.completion import AzureCompletion

        comp = AzureCompletion(
            model="gpt-4o",
            api_key="key",
            endpoint="https://res.openai.azure.com",
        )
        assert comp.last_response_id is None

    def test_last_reasoning_items(self, mock_openai_completion):
        _mock_cls, _ = mock_openai_completion
        comp = _create_azure_responses()
        assert comp.last_reasoning_items == [{"type": "reasoning"}]

    def test_reset_chain(self, mock_openai_completion):
        _mock_cls, instance = mock_openai_completion
        comp = _create_azure_responses()
        comp.reset_chain()
        instance.reset_chain.assert_called_once()

    def test_reset_reasoning_chain(self, mock_openai_completion):
        _mock_cls, instance = mock_openai_completion
        comp = _create_azure_responses()
        comp.reset_reasoning_chain()
        instance.reset_reasoning_chain.assert_called_once()

    def test_reset_chain_noop_for_completions(self):
        """reset_chain should not raise when delegate is None."""
        from crewai.llms.providers.azure.completion import AzureCompletion

        comp = AzureCompletion(
            model="gpt-4o",
            api_key="key",
            endpoint="https://res.openai.azure.com",
        )
        comp.reset_chain()  # should not raise


# ---------------------------------------------------------------------------
# Feature-support method tests
# ---------------------------------------------------------------------------


class TestAzureResponsesFeatures:
    """Test supports_* and config methods."""

    def test_supports_function_calling_responses(self, mock_openai_completion):
        _mock_cls, _ = mock_openai_completion
        comp = _create_azure_responses()
        assert comp.supports_function_calling() is True

    def test_supports_function_calling_completions_openai_model(self):
        from crewai.llms.providers.azure.completion import AzureCompletion

        comp = AzureCompletion(
            model="gpt-4o",
            api_key="key",
            endpoint="https://res.openai.azure.com",
        )
        assert comp.supports_function_calling() is True

    def test_supports_stop_words_false_for_responses(self, mock_openai_completion):
        _mock_cls, _ = mock_openai_completion
        comp = _create_azure_responses(model="o4-mini")
        assert comp.supports_stop_words() is False

    def test_supports_stop_words_true_for_completions_gpt4(self):
        from crewai.llms.providers.azure.completion import AzureCompletion

        comp = AzureCompletion(
            model="gpt-4o",
            api_key="key",
            endpoint="https://res.openai.azure.com",
        )
        assert comp.supports_stop_words() is True

    def test_to_config_dict_includes_responses_fields(self, mock_openai_completion):
        _mock_cls, _ = mock_openai_completion
        comp = _create_azure_responses(
            reasoning_effort="high",
            instructions="Be concise",
            store=True,
            max_completion_tokens=500,
        )
        config = comp.to_config_dict()
        assert config["api"] == "responses"
        assert config["reasoning_effort"] == "high"
        assert config["instructions"] == "Be concise"
        assert config["store"] is True
        assert config["max_completion_tokens"] == 500

    def test_to_config_dict_omits_api_for_completions(self):
        from crewai.llms.providers.azure.completion import AzureCompletion

        comp = AzureCompletion(
            model="gpt-4o",
            api_key="key",
            endpoint="https://res.openai.azure.com",
        )
        config = comp.to_config_dict()
        assert "api" not in config


# ---------------------------------------------------------------------------
# LLM factory integration test
# ---------------------------------------------------------------------------


class TestAzureResponsesViaLLMFactory:
    """Test that the LLM factory passes api='responses' through to AzureCompletion."""

    @pytest.mark.usefixtures("azure_env")
    def test_llm_factory_passes_api_kwarg(self):
        """LLM(model='azure/gpt-4o', api='responses') should create AzureCompletion
        with api='responses' and a delegate."""
        with (
            patch(
                "crewai.llms.providers.openai.completion.OpenAI",
            ),
            patch(
                "crewai.llms.providers.openai.completion.AsyncOpenAI",
            ),
        ):
            from crewai.llm import LLM

            llm = LLM(model="azure/gpt-4o", api="responses")

            from crewai.llms.providers.azure.completion import AzureCompletion

            assert isinstance(llm, AzureCompletion)
            assert llm.api == "responses"
            assert llm._responses_delegate is not None
