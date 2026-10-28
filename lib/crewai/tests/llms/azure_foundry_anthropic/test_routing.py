import pytest

pytest.importorskip("anthropic")

from crewai.llm import LLM
from crewai.llms.providers.azure.completion import AzureCompletion
from crewai.llms.providers.azure_foundry_anthropic.completion import (
    AzureFoundryAnthropicCompletion,
)


@pytest.mark.parametrize(
    "model",
    [
        "azure/claude-sonnet-4-6",
        "azure/claude-sonnet-4-6-2",
        "azure/Claude-Opus-4-6",
        "azure/anthropic-sonnet-prod",
    ],
)
def test_azure_claude_models_route_to_foundry_class(model: str) -> None:
    route = LLM._resolve_route(model, {})

    assert route.native_class is AzureFoundryAnthropicCompletion
    assert route.provider == "azure"


@pytest.mark.parametrize(
    "model", ["azure/gpt-4.1", "azure/gpt-4o", "azure/o3-mini", "azure/vision-prod"]
)
def test_other_azure_models_stay_on_azure_completion(model: str) -> None:
    assert LLM._resolve_route(model, {}).native_class is AzureCompletion


def test_explicit_azure_provider_with_claude_model_routes_to_foundry_class() -> None:
    route = LLM._resolve_route("claude-sonnet-4-6", {"provider": "azure"})

    assert route.native_class is AzureFoundryAnthropicCompletion


def test_direct_anthropic_route_is_unchanged() -> None:
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    route = LLM._resolve_route("anthropic/claude-sonnet-4-6", {})

    assert route.native_class is AnthropicCompletion


def test_foundry_class_is_built_with_azure_provider() -> None:
    llm = LLM(model="azure/claude-sonnet-4-6", api_key="test-key")

    assert type(llm) is AzureFoundryAnthropicCompletion
    assert llm.provider == "azure"
    assert llm.model == "claude-sonnet-4-6"


@pytest.mark.parametrize(
    "model", ["azure/claude-sonnet-4-6", "azure/azure_ai/claude-sonnet-4-6"]
)
def test_claude_deployments_support_multimodal(model: str) -> None:
    llm = LLM(model=model, api_key="test-key")

    assert type(llm) is AzureFoundryAnthropicCompletion
    assert llm.supports_multimodal() is True


@pytest.mark.parametrize("override", [True, False])
def test_supports_vision_override_applies_to_claude_deployments(
    override: bool,
) -> None:
    llm = LLM(model="azure/claude-prod", api_key="test-key", supports_vision=override)

    assert type(llm) is AzureFoundryAnthropicCompletion
    assert llm.supports_multimodal() is override
