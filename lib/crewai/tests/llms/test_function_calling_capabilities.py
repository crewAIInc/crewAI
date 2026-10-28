from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.azure.completion import AzureCompletion
from crewai.llms.providers.bedrock.completion import BedrockCompletion
from crewai.llms.providers.gemini.completion import GeminiCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.llms.providers.openai_compatible.completion import (
    OpenAICompatibleCompletion,
)
from crewai.llms.providers.snowflake.completion import SnowflakeCompletion


def _set_catalog_entry(llm: Any, entry: dict[str, Any] | None) -> None:
    llm._catalog_entry = entry
    llm._catalog_entry_resolved = True


@pytest.mark.parametrize(
    ("factory", "expected_fallback"),
    [
        (lambda: OpenAICompletion(model="o1-mini", api_key="test-key"), False),
        (lambda: OpenAICompletion(model="gpt-4o", api_key="test-key"), True),
        (
            lambda: OpenAICompatibleCompletion(
                model="o1-mini", provider="ollama"
            ),
            False,
        ),
        (
            lambda: OpenAICompatibleCompletion(model="llama3", provider="ollama"),
            True,
        ),
        (lambda: AzureCompletion(model="deepseek-chat"), False),
        (lambda: AzureCompletion(model="gpt-4"), True),
        (
            lambda: AnthropicCompletion(
                model="claude-3-5-sonnet-20241022", supports_tools=False
            ),
            False,
        ),
        (lambda: AnthropicCompletion(model="claude-3-5-sonnet-20241022"), True),
        (
            lambda: BedrockCompletion(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                supports_tools=False,
            ),
            False,
        ),
        (
            lambda: BedrockCompletion(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0"
            ),
            True,
        ),
        (lambda: GeminiCompletion(model="gemini-1.0-pro"), False),
        (lambda: GeminiCompletion(model="gemini-1.5-pro"), True),
        (
            lambda: SnowflakeCompletion(
                model="mistral-large", api_key="test-token", account_url="example"
            ),
            False,
        ),
        (
            lambda: SnowflakeCompletion(
                model="openai-gpt-4o", api_key="test-token", account_url="example"
            ),
            True,
        ),
    ],
)
def test_function_calling_catalog_boolean_overrides_provider_fallback(
    factory: Callable[[], Any], expected_fallback: bool
) -> None:
    llm = factory()
    _set_catalog_entry(llm, None)

    assert llm.supports_function_calling() is expected_fallback

    _set_catalog_entry(llm, {"supports_function_calling": True})
    assert llm.supports_function_calling() is True

    _set_catalog_entry(llm, {"supports_function_calling": False})
    assert llm.supports_function_calling() is False


def test_litellm_function_calling_falls_back_when_catalog_field_is_missing() -> None:
    llm = OpenAICompletion(model="custom-model", api_key="test-key")
    _set_catalog_entry(llm, {})
    llm._get_custom_llm_provider = lambda: "openai"

    fallback = SimpleNamespace(
        utils=SimpleNamespace(supports_function_calling=lambda *_args, **_kwargs: False)
    )
    with (
        patch("crewai.llm._ensure_litellm", return_value=True),
        patch("crewai.llm.litellm", fallback),
    ):
        assert LLM.supports_function_calling(llm) is False


@pytest.mark.parametrize("catalog_support", [True, False])
def test_litellm_catalog_function_calling_boolean_is_authoritative(
    catalog_support: bool,
) -> None:
    llm = OpenAICompletion(model="custom-model", api_key="test-key")
    _set_catalog_entry(llm, {"supports_function_calling": catalog_support})
    llm._get_custom_llm_provider = lambda: "openai"

    fallback = SimpleNamespace(utils=SimpleNamespace(supports_function_calling=None))
    with (
        patch("crewai.llm._ensure_litellm", return_value=True),
        patch("crewai.llm.litellm", fallback),
    ):
        assert LLM.supports_function_calling(llm) is catalog_support

    assert fallback.utils.supports_function_calling is None


@pytest.mark.parametrize("catalog_support", [True, False])
def test_anthropic_request_tool_gate_uses_catalog_capability(
    catalog_support: bool,
) -> None:
    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022")
    _set_catalog_entry(llm, {"supports_function_calling": catalog_support})

    params = llm._prepare_completion_params(
        [{"role": "user", "content": "hello"}],
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
    )

    assert ("tools" in params) is catalog_support


@pytest.mark.parametrize("catalog_support", [True, False])
def test_gemini_request_tool_gate_uses_catalog_capability(
    catalog_support: bool,
) -> None:
    llm = GeminiCompletion(model="gemini-1.0-pro")
    _set_catalog_entry(llm, {"supports_function_calling": catalog_support})

    config = llm._prepare_generation_config(
        tools=[{"name": "get_weather", "parameters": {"type": "object"}}]
    )

    assert (getattr(config, "tools", None) is not None) is catalog_support


@pytest.mark.parametrize("catalog_support", [True, False])
def test_bedrock_request_tool_gate_uses_catalog_capability(
    catalog_support: bool,
) -> None:
    llm = BedrockCompletion(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
    _set_catalog_entry(llm, {"supports_function_calling": catalog_support})
    llm._client = MagicMock()
    llm._client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }

    llm.call(
        "hello",
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object"},
            }
        ],
    )

    request = llm._client.converse.call_args.kwargs
    assert ("toolConfig" in request) is catalog_support
