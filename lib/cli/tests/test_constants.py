from importlib.metadata import metadata

import pytest

from crewai_cli.constants import (
    ENV_VARS,
    MODELS,
    PROVIDER_EXTRAS,
    PROVIDERS,
    provider_for_model,
)


def test_huggingface_in_providers():
    """Test that Huggingface is in the PROVIDERS list."""
    assert "huggingface" in PROVIDERS


def test_huggingface_env_vars():
    """Test that Huggingface environment variables are properly configured."""
    assert "huggingface" in ENV_VARS
    assert any(
        detail.get("key_name") == "HF_TOKEN" for detail in ENV_VARS["huggingface"]
    )


def test_huggingface_models():
    """Test that Huggingface models are properly configured."""
    assert "huggingface" in MODELS
    assert len(MODELS["huggingface"]) > 0


def test_openai_models_include_gpt56_family() -> None:
    """Curated OpenAI list should include Sol, Terra, Luna, and the alias."""
    openai_models = MODELS["openai"]
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6"):
        assert model in openai_models


def test_gemini_models_offer_no_shut_down_model() -> None:
    """`crewai create` must not offer models Google has shut down; picking one
    fails the first run with a ClientError."""
    gemini_models = MODELS["gemini"]
    assert gemini_models[0] == "gemini/gemini-3.8-flash"
    for model in (
        "gemini/gemini-4-argon",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.5-flash-lite",
    ):
        assert model in gemini_models
    shut_down = {
        "gemini/gemini-3-pro-preview",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-2.0-flash-lite-001",
        "gemini/gemini-2.0-flash-001",
        "gemini/gemini-2.0-flash-thinking-exp-01-21",
        "gemini/gemini-2.5-flash-preview-04-17",
        "gemini/gemini-2.5-pro-exp-03-25",
    }
    assert not shut_down.intersection(gemini_models)


def test_default_wizard_offers_no_gemini_model_new_accounts_cannot_use() -> None:
    """With no key, `crewai create crew` shows the curated Gemini list, so it
    must hold only models a new account can call; the first is the default."""
    from crewai_cli.create_json_crew import (
        _PROVIDER_MODELS,
        _default_model_for_provider,
    )

    gemini_models = [model_id for model_id, _ in _PROVIDER_MODELS["gemini"]]
    assert _default_model_for_provider("gemini") == "gemini/gemini-3.8-flash"
    for model in ("gemini-4-argon", "gemini-3.6-flash", "gemini-3.5-flash-lite"):
        assert model in gemini_models
    assert not [m for m in gemini_models if m.startswith(("gemini-2.", "gemini-1."))]


def test_classic_lists_lead_with_current_small_models() -> None:
    """`crewai create crew --classic` lists models in MODELS order."""
    assert MODELS["openai"][0] == "gpt-5.6-luna"
    assert "gpt-5.4-mini" not in MODELS["openai"]
    assert MODELS["anthropic"][0] == "claude-sonnet-5"


def test_wizard_lists_lead_with_current_small_models() -> None:
    """With no key, the wizard shows the curated list in order; the first is the
    `--provider` default."""
    from crewai_cli.create_json_crew import _PROVIDER_MODELS

    assert [model_id for model_id, _ in _PROVIDER_MODELS["openai"]] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.2",
        "gpt-4.1",
    ]
    assert [model_id for model_id, _ in _PROVIDER_MODELS["anthropic"]] == [
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "claude-opus-4-8",
        "claude-fable-5",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
    ]


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("openai/gpt-5.6-luna", "openai"),
        ("anthropic/claude-sonnet-5", "anthropic"),
        ("gemini/gemini-3.8-flash", "gemini"),
        ("bedrock/us.amazon.nova-pro-v1:0", "bedrock"),
        ("azure/my-deployment", "azure"),
        ("groq/meta-llama/llama-4-scout-17b-16e-instruct", "groq"),
        ("ollama/llama3.3", "ollama"),
        ("watson/ibm/granite-3-8b-instruct", "watson"),
        ("Anthropic/claude-sonnet-5", "anthropic"),
        ("  openai/gpt-5.6-luna  ", "openai"),
    ],
)
def test_provider_for_model_reads_the_prefix(model: str, provider: str) -> None:
    assert provider_for_model(model) == provider


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude/claude-sonnet-5", "anthropic"),
        ("google/gemini-3.8-flash", "gemini"),
        ("aws/us.amazon.nova-pro-v1:0", "bedrock"),
        ("azure_openai/gpt-4o", "azure"),
    ],
)
def test_provider_for_model_applies_crewai_prefix_aliases(
    model: str, provider: str
) -> None:
    assert provider_for_model(model) == provider


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gpt-5.6-luna", "openai"),
        ("o1", "openai"),
        ("o3-mini", "openai"),
        ("o4-mini", "openai"),
        ("chatgpt-4o-latest", "openai"),
        ("claude-sonnet-5", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("gemini-3.8-flash", "gemini"),
    ],
)
def test_provider_for_model_recognizes_bare_model_names(
    model: str, provider: str
) -> None:
    assert provider_for_model(model) == provider


@pytest.mark.parametrize(
    "model", ["", "   ", "llama3.3", "my-model", "mistral-large", "/gpt-5.6-luna"]
)
def test_provider_for_model_returns_none_for_an_unknown_model(model: str) -> None:
    assert provider_for_model(model) is None


def test_every_provider_extra_is_a_crewai_extra() -> None:
    """A name crewai does not publish makes `uv sync` warn and install nothing."""
    published = set(metadata("crewai").get_all("Provides-Extra") or [])

    assert set(PROVIDER_EXTRAS.values()) <= published


def test_every_provider_create_offers_is_mapped_or_needs_no_extra() -> None:
    """A provider added to either picker has to be checked against crewai's
    routing: openai, ollama and cerebras run on `crewai[tools]` alone."""
    from crewai_cli.create_json_crew import _PROVIDERS

    offered = set(PROVIDERS) | {key for key, _ in _PROVIDERS}

    assert offered - set(PROVIDER_EXTRAS) == {"openai", "ollama", "cerebras"}
    assert set(PROVIDER_EXTRAS) <= offered


def test_provider_for_model_matches_every_prefix_crewai_routes() -> None:
    """The CLI keeps its own copy of crewAI's prefix aliases; a prefix crewAI
    adds or renames must not make `crewai create` pick the wrong extra."""
    from crewai.llm import PROVIDER_ALIASES

    for prefix, provider in PROVIDER_ALIASES.items():
        assert provider_for_model(f"{prefix}/some-model") == provider, prefix
