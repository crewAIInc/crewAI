import pytest

pytest.importorskip("anthropic")

from crewai.llm import LLM
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.azure.completion import AzureCompletion
from crewai.llms.providers.azure_foundry_anthropic.completion import (
    AzureFoundryAnthropicCompletion,
)
from crewai.utilities.llm_utils import create_llm_like
from crewai_files import get_constraints_for_provider, get_supported_content_types

FOUNDRY = "https://res.services.ai.azure.com"


@pytest.fixture(autouse=True)
def azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_ENDPOINT", "https://env.services.ai.azure.com")
    monkeypatch.setenv("AZURE_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    monkeypatch.delenv("AZURE_API_BASE", raising=False)


class TestGenericAzureIsUnchanged:
    def test_gpt_deployment_still_builds_azure_completion(self) -> None:
        llm = LLM(model="azure/gpt-4.1")

        assert type(llm) is AzureCompletion
        assert not isinstance(llm, AnthropicCompletion)

    def test_file_formatting_still_uses_azure_constraints(self) -> None:
        llm = LLM(model="azure/gpt-4.1")

        assert llm._multimodal_formatter_name() == "azure"
        assert "application/pdf" not in get_supported_content_types("azure")
        assert get_constraints_for_provider("azure").supports_file_upload is False

    def test_openai_host_endpoint_does_not_break_building_a_claude_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_ENDPOINT", "https://res.openai.azure.com")

        llm = LLM(model="azure/claude-sonnet-4-6")

        assert type(llm) is AzureFoundryAnthropicCompletion


class TestDirectAnthropicIsUnchanged:
    def test_anthropic_prefix_still_builds_anthropic_completion(self) -> None:
        llm = LLM(model="anthropic/claude-sonnet-4-6")

        assert type(llm) is AnthropicCompletion
        assert llm.provider == "anthropic"

    def test_file_formatting_still_uses_anthropic_constraints(self) -> None:
        llm = LLM(model="anthropic/claude-sonnet-4-6")

        assert llm._multimodal_formatter_name() == "anthropic"
        assert "application/pdf" in get_supported_content_types("anthropic")
        assert get_constraints_for_provider("anthropic").supports_file_upload is True

    def test_direct_anthropic_keeps_its_own_key(self) -> None:
        llm = LLM(model="anthropic/claude-sonnet-4-6")

        assert llm._get_client_params()["api_key"] == "direct-key"


class TestRoutingRule:
    @pytest.mark.parametrize(
        "model", ["azure/my-claude-prod", "azure/CLAUDE-opus", "azure/anthropic-x"]
    )
    def test_a_marker_anywhere_in_the_name_routes_to_foundry(self, model: str) -> None:
        assert LLM._resolve_route(model, {}).native_class is (
            AzureFoundryAnthropicCompletion
        )

    def test_azure_openai_alias_routes_to_foundry_and_keeps_its_provider(self) -> None:
        route = LLM._resolve_route("claude-sonnet-4-6", {"provider": "azure_openai"})

        assert route.native_class is AzureFoundryAnthropicCompletion
        assert route.provider == "azure_openai"

    def test_other_providers_with_claude_models_are_not_rerouted(self) -> None:
        for model in ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"):
            assert LLM._resolve_route(model, {}).native_class is AnthropicCompletion


class TestSwappingModels:
    def test_swap_from_azure_gpt_to_claude_keeps_settings_not_credentials(
        self,
    ) -> None:
        base = LLM(
            model="azure/gpt-4.1",
            endpoint="https://r.openai.azure.com",
            api_key="gpt-key",
            temperature=0.2,
        )

        swapped = create_llm_like("azure/claude-sonnet-4-6", base)

        assert type(swapped) is AzureFoundryAnthropicCompletion
        assert swapped.temperature == 0.2
        assert swapped.endpoint == "https://env.services.ai.azure.com"
        assert swapped._get_client_params()["api_key"] == "env-key"

    def test_swap_from_claude_to_azure_gpt_does_not_carry_the_foundry_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_ENDPOINT", "https://env.openai.azure.com")
        base = LLM(
            model="azure/claude-sonnet-4-6", endpoint=FOUNDRY, api_key="foundry-key"
        )

        swapped = create_llm_like("azure/gpt-4.1", base)

        assert type(swapped) is AzureCompletion
        assert "env.openai.azure.com" in swapped.endpoint
        assert swapped.api_key == "env-key"

    def test_swap_between_claude_models_keeps_endpoint_and_key(self) -> None:
        base = LLM(
            model="azure/claude-sonnet-4-6", endpoint=FOUNDRY, api_key="foundry-key"
        )

        swapped = create_llm_like("azure/claude-opus-4-6", base)

        assert type(swapped) is AzureFoundryAnthropicCompletion
        assert swapped.endpoint == FOUNDRY
        assert swapped.api_key == "foundry-key"


def test_model_copy_keeps_class_provider_and_endpoint() -> None:
    llm = LLM(model="azure/claude-sonnet-4-6", endpoint=FOUNDRY, api_key="k")

    copied = llm.model_copy()

    assert type(copied) is AzureFoundryAnthropicCompletion
    assert copied.provider == "azure"
    assert copied.endpoint == FOUNDRY
