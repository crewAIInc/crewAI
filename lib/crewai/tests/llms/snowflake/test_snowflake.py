from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.snowflake.completion import (
    SNOWFLAKE_CORTEX_PATH,
    SnowflakeCompletion,
    _normalize_snowflake_base_url,
)


def _snowflake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_PAT", "test-pat")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT_URL", "https://org-account.snowflakecomputing.com")
    monkeypatch.delenv("SNOWFLAKE_TOKEN", raising=False)
    monkeypatch.delenv("SNOWFLAKE_JWT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT_IDENTIFIER", raising=False)


class TestSnowflakeConfig:
    def test_normalizes_account_url_to_cortex_base_url(self):
        assert (
            _normalize_snowflake_base_url("https://org-account.snowflakecomputing.com")
            == f"https://org-account.snowflakecomputing.com{SNOWFLAKE_CORTEX_PATH}"
        )

    def test_preserves_existing_cortex_base_url(self):
        base_url = f"https://org-account.snowflakecomputing.com{SNOWFLAKE_CORTEX_PATH}"
        assert _normalize_snowflake_base_url(base_url) == base_url

    def test_rejects_endpoint_path_in_base_url(self):
        with pytest.raises(ValueError, match="do not include endpoint paths"):
            _normalize_snowflake_base_url(
                "https://org-account.snowflakecomputing.com"
                f"{SNOWFLAKE_CORTEX_PATH}/chat/completions"
            )

    def test_empty_api_key_falls_back_to_env_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)

        llm = SnowflakeCompletion(model="openai-gpt-4.1", api_key="")

        assert llm.api_key == "test-pat"

    def test_uses_env_token_and_account_url(self, monkeypatch: pytest.MonkeyPatch):
        _snowflake_env(monkeypatch)

        llm = SnowflakeCompletion(model="openai-gpt-4.1")

        assert llm.api_key == "test-pat"
        assert llm.base_url == (
            f"https://org-account.snowflakecomputing.com{SNOWFLAKE_CORTEX_PATH}"
        )
        assert llm.account_url == llm.base_url

    def test_strips_litellm_pat_prefix_for_compatibility(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SNOWFLAKE_PAT", "pat/test-pat")
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "org-account")

        llm = SnowflakeCompletion(model="openai-gpt-4.1")

        assert llm.api_key == "test-pat"

    def test_missing_token_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SNOWFLAKE_PAT", raising=False)
        monkeypatch.delenv("SNOWFLAKE_TOKEN", raising=False)
        monkeypatch.delenv("SNOWFLAKE_JWT", raising=False)
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT_URL", "https://org-account.snowflakecomputing.com")

        with pytest.raises(ValueError, match="Snowflake token is required"):
            SnowflakeCompletion(model="openai-gpt-4.1")

    def test_missing_account_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SNOWFLAKE_PAT", "test-pat")
        monkeypatch.delenv("SNOWFLAKE_ACCOUNT_URL", raising=False)
        monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
        monkeypatch.delenv("SNOWFLAKE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("SNOWFLAKE_ACCOUNT_IDENTIFIER", raising=False)

        with pytest.raises(ValueError, match="Snowflake account URL is required"):
            SnowflakeCompletion(model="openai-gpt-4.1")

    def test_responses_api_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        _snowflake_env(monkeypatch)

        with pytest.raises(ValueError, match="supports only the Chat Completions API"):
            SnowflakeCompletion(model="openai-gpt-4.1", api="responses")


class TestSnowflakeFactory:
    def test_llm_creates_native_snowflake_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)

        llm = LLM(model="snowflake/openai-gpt-4.1")

        assert isinstance(llm, SnowflakeCompletion)
        assert llm.provider == "snowflake"
        assert llm.model == "openai-gpt-4.1"
        assert llm.is_litellm is False

    def test_explicit_provider_creates_native_snowflake_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)

        llm = LLM(model="claude-sonnet-4-5", provider="snowflake")

        assert isinstance(llm, SnowflakeCompletion)
        assert llm.model == "claude-sonnet-4-5"


class TestSnowflakeRequests:
    def test_prepare_completion_params_uses_snowflake_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)
        llm = SnowflakeCompletion(
            model="openai-gpt-4.1",
            temperature=0.2,
            max_completion_tokens=128,
        )

        params = llm._prepare_completion_params(
            [{"role": "user", "content": "Hello"}]
        )

        assert params["model"] == "openai-gpt-4.1"
        assert params["temperature"] == 0.2
        assert params["max_completion_tokens"] == 128
        assert params["messages"] == [{"role": "user", "content": "Hello"}]

    def test_claude_model_removes_trailing_assistant_prefill(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)
        llm = SnowflakeCompletion(model="claude-sonnet-4-5")

        messages = llm._format_messages(
            [
                {"role": "user", "content": "Write a summary."},
                {"role": "assistant", "content": "Here is"},
            ]
        )

        assert messages == [{"role": "user", "content": "Write a summary."}]

    def test_claude_model_adds_user_turn_after_tool_call_assistant_message(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)
        llm = SnowflakeCompletion(model="claude-sonnet-4-5")

        messages = llm._format_messages(
            [
                {"role": "user", "content": "Use the tool."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )

        assert messages[-2]["role"] == "assistant"
        assert messages[-2]["tool_calls"][0]["id"] == "call_1"
        assert messages[-1]["role"] == "user"

    def test_claude_model_maps_max_tokens_to_max_completion_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)
        llm = SnowflakeCompletion(model="claude-sonnet-4-5", max_tokens=256)

        params = llm._prepare_completion_params(
            [{"role": "user", "content": "Hello"}]
        )

        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 256

    def test_streaming_params_include_usage(self, monkeypatch: pytest.MonkeyPatch):
        _snowflake_env(monkeypatch)
        llm = SnowflakeCompletion(model="openai-gpt-4.1", stream=True)

        params = llm._prepare_completion_params(
            [{"role": "user", "content": "Hello"}]
        )

        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": True}

    def test_non_streaming_call_uses_native_openai_client(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _snowflake_env(monkeypatch)
        llm = SnowflakeCompletion(model="openai-gpt-4.1")
        fake_response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Snowflake response", tool_calls=None)
                )
            ],
        )
        create = Mock(return_value=fake_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch.object(llm, "_get_sync_client", return_value=fake_client):
            response = llm.call([{"role": "user", "content": "Hello"}])

        assert response == "Snowflake response"
        create.assert_called_once()
        assert create.call_args.kwargs["model"] == "openai-gpt-4.1"
        assert create.call_args.kwargs["messages"] == [
            {"role": "user", "content": "Hello"}
        ]
