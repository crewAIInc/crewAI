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


@pytest.fixture
def stub_catalog(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any] | None], None]:
    """Make every LLM created afterwards resolve ``entry`` as its catalog entry."""

    def stub(entry: dict[str, Any] | None) -> None:
        monkeypatch.setattr(
            "crewai.llms.base_llm.find_model_catalog_entry",
            lambda model, provider=None: entry,
        )

    return stub


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
        (lambda: AnthropicCompletion(model="claude-3-5-sonnet-20241022"), True),
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
    factory: Callable[[], Any],
    expected_fallback: bool,
    stub_catalog: Callable[[dict[str, Any] | None], None],
) -> None:
    stub_catalog(None)
    assert factory().supports_function_calling() is expected_fallback

    stub_catalog({"supports_function_calling": True})
    assert factory().supports_function_calling() is True

    stub_catalog({"supports_function_calling": False})
    assert factory().supports_function_calling() is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AnthropicCompletion(
            model="claude-3-5-sonnet-20241022", supports_tools=False
        ),
        lambda: BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0", supports_tools=False
        ),
    ],
)
def test_explicit_supports_tools_false_overrides_catalog(
    factory: Callable[[], Any],
    stub_catalog: Callable[[dict[str, Any] | None], None],
) -> None:
    stub_catalog(None)
    assert factory().supports_function_calling() is False

    stub_catalog({"supports_function_calling": True})
    assert factory().supports_function_calling() is False


def test_litellm_function_calling_falls_back_when_catalog_field_is_missing(stub_catalog: Callable[[dict[str, Any] | None], None]) -> None:
    stub_catalog({})
    llm = OpenAICompletion(model="custom-model", api_key="test-key")
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
def test_litellm_catalog_function_calling_boolean_is_authoritative(catalog_support: bool, stub_catalog: Callable[[dict[str, Any] | None], None]) -> None:
    stub_catalog({"supports_function_calling": catalog_support})
    llm = OpenAICompletion(model="custom-model", api_key="test-key")
    llm._get_custom_llm_provider = lambda: "openai"

    fallback = SimpleNamespace(utils=SimpleNamespace(supports_function_calling=None))
    with (
        patch("crewai.llm._ensure_litellm", return_value=True),
        patch("crewai.llm.litellm", fallback),
    ):
        assert LLM.supports_function_calling(llm) is catalog_support

    assert fallback.utils.supports_function_calling is None


@pytest.mark.parametrize("catalog_support", [True, False])
def test_anthropic_request_tool_gate_uses_catalog_capability(catalog_support: bool, stub_catalog: Callable[[dict[str, Any] | None], None]) -> None:
    stub_catalog({"supports_function_calling": catalog_support})
    llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022")

    params = llm._prepare_completion_params(
        [{"role": "user", "content": "hello"}],
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
    )

    assert ("tools" in params) is catalog_support


@pytest.mark.parametrize("catalog_support", [True, False])
def test_gemini_request_tool_gate_uses_catalog_capability(catalog_support: bool, stub_catalog: Callable[[dict[str, Any] | None], None]) -> None:
    stub_catalog({"supports_function_calling": catalog_support})
    llm = GeminiCompletion(model="gemini-1.0-pro")

    config = llm._prepare_generation_config(
        tools=[{"name": "get_weather", "parameters": {"type": "object"}}]
    )

    assert (getattr(config, "tools", None) is not None) is catalog_support


@pytest.mark.parametrize("catalog_support", [True, False])
def test_bedrock_request_tool_gate_uses_catalog_capability(catalog_support: bool, stub_catalog: Callable[[dict[str, Any] | None], None]) -> None:
    stub_catalog({"supports_function_calling": catalog_support})
    llm = BedrockCompletion(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
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


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            lambda: OpenAICompletion(
                model="gpt-4o", provider="openai", api_key="test-key"
            ),
            True,
        ),
        (
            lambda: OpenAICompletion(
                model="gpt-5-chat", provider="openai", api_key="test-key"
            ),
            False,
        ),
        (lambda: AzureCompletion(model="gpt-4o", provider="azure"), True),
        (
            lambda: AnthropicCompletion(
                model="claude-sonnet-4-5", provider="anthropic"
            ),
            True,
        ),
        (
            lambda: BedrockCompletion(
                model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                provider="bedrock",
            ),
            True,
        ),
        (
            lambda: GeminiCompletion(model="gemini-2.5-flash", provider="gemini"),
            True,
        ),
    ],
)
def test_catalog_entry_resolves_through_the_real_lookup(
    factory: Callable[[], Any], expected: bool
) -> None:
    llm = factory()

    assert llm.catalog_entry is not None
    assert llm.catalog_entry["supports_function_calling"] is expected
    assert llm.supports_function_calling() is expected


@pytest.mark.parametrize(
    ("model", "expected_class", "expected_provider"),
    [
        ("azure/gpt-4o", AzureCompletion, "azure"),
        ("anthropic/claude-sonnet-4-5", AnthropicCompletion, "anthropic"),
        (
            "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            BedrockCompletion,
            "bedrock",
        ),
    ],
)
def test_llm_factory_resolves_the_catalog_under_the_routed_provider(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_class: type,
    expected_provider: str,
) -> None:
    for key in (
        "AZURE_API_KEY",
        "AZURE_ENDPOINT",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.setenv(
            key, "https://example.openai.azure.com" if key == "AZURE_ENDPOINT" else "x"
        )

    llm = LLM(model=model)

    assert isinstance(llm, expected_class)
    assert llm.provider == expected_provider
    assert llm.catalog_entry is not None
    assert llm.catalog_entry["litellm_provider"].startswith(expected_provider)
