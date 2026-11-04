from importlib.metadata import metadata

import pytest

from crewai_cli.constants import (
    ENV_VARS,
    MODELS,
    PROVIDER_EXTRAS,
    PROVIDERS,
    crewai_extras_for_model,
    provider_for_model,
    required_key_names,
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
        ("gemma-3-27b-it", "gemini"),
        ("learnlm-2.0-flash", "gemini"),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock"),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock"),
    ],
)
def test_provider_for_model_recognizes_bare_model_names(
    model: str, provider: str
) -> None:
    assert provider_for_model(model) == provider


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.6-luna",
        "o3-mini",
        "claude-sonnet-5",
        "gemini-3.8-flash",
        "gemma-3-27b-it",
        "learnlm-2.0-flash",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
def test_provider_for_model_infers_bare_names_as_crewai_does(model: str) -> None:
    """A bare name the CLI recognizes must land on the provider crewAI infers,
    or `crewai create` adds the wrong extra and the first run fails."""
    from crewai.llm import LLM

    assert provider_for_model(model) == LLM._infer_provider_from_model(model)


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


@pytest.mark.parametrize(
    ("model", "extras"),
    [
        ("anthropic/claude-sonnet-5", {"anthropic"}),
        ("claude/claude-sonnet-5", {"anthropic"}),
        ("gemini/gemma-3-27b-it", {"google-genai"}),
        ("bedrock/us.amazon.nova-pro-v1:0", {"bedrock"}),
        ("bedrock/us-gov-east-1/anthropic.claude-haiku-4-5-20251001-v1:0", {"bedrock"}),
        ("azure/my-deployment", {"azure-ai-inference"}),
        ("groq/llama-3.3-70b-versatile", {"litellm"}),
        ("openai/gpt-5.6-luna", set()),
        ("ollama/llama3.3", set()),
        ("cerebras/llama3.1-8b", set()),
        ("deepseek/deepseek-chat", set()),
    ],
)
def test_crewai_extras_for_a_model_its_provider_runs_natively(
    model: str, extras: set[str]
) -> None:
    assert crewai_extras_for_model(model) == extras


@pytest.mark.parametrize(
    ("model", "extras"),
    [
        ("bedrock/claude-sonnet-4-5-20250929-v1:0", {"bedrock", "litellm"}),
        ("aws/us-gov-west-1/claude-sonnet-4-5-20250929-v1:0", {"bedrock", "litellm"}),
        ("anthropic/my-finetune", {"anthropic", "litellm"}),
        ("claude/my-finetune", {"anthropic", "litellm"}),
        ("gemini/aqa", {"google-genai", "litellm"}),
        ("openai/chatgpt-4o-latest", {"litellm"}),
        ("OpenAI/codex-mini-latest", {"litellm"}),
        ("deepseek/my-model", {"litellm"}),
    ],
)
def test_crewai_extras_add_litellm_for_a_name_the_native_provider_does_not_take(
    model: str, extras: set[str]
) -> None:
    assert crewai_extras_for_model(model) == extras


@pytest.mark.parametrize(
    ("model", "extras"),
    [
        ("chatgpt-4o-latest", set()),
        ("gpt-5.6-luna", set()),
        ("claude-sonnet-5", {"anthropic"}),
        ("gemini-3.8-flash", {"google-genai"}),
        ("my-model", set()),
        ("custom/my-model", set()),
        ("", set()),
    ],
)
def test_crewai_extras_for_a_bare_or_unknown_model(
    model: str, extras: set[str]
) -> None:
    assert crewai_extras_for_model(model) == extras


_SAMPLE_MODEL_NAMES = [
    "gpt-5.6-luna",
    "o4-mini",
    "chatgpt-4o-latest",
    "whisper-1",
    "claude-sonnet-5",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5-20250929-v1:0",
    "us-gov-east-1/claude-sonnet-4-5-20250929-v1:0",
    "us.amazon.nova-pro-v1:0",
    "gemini-3.8-flash",
    "gemma-3-27b-it",
    "learnlm-2.0-flash",
    "aqa",
    "deepseek-chat",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "my-finetune",
]


def test_litellm_extra_matches_crewai_routing_for_every_native_prefix() -> None:
    """The CLI copies crewAI's rule for which names a native provider takes; a
    name crewAI sends to LiteLLM must get the litellm extra, and no other."""
    from crewai.llm import LLM, PROVIDER_ALIASES

    for prefix in PROVIDER_ALIASES:
        if prefix == "deepinfra":
            # crewAI raises on an id that is not org/model instead of routing it.
            continue
        for name in _SAMPLE_MODEL_NAMES:
            model = f"{prefix}/{name}"
            routed_to_litellm = LLM._resolve_route(model, {}).native_class is None
            assert ("litellm" in crewai_extras_for_model(model)) is (
                routed_to_litellm
            ), model


# Bedrock has no curated wizard list, so its picker shows the LiteLLM feed.
# These are the ids the feed put there on 2026-11-03.
_BEDROCK_FEED_MODELS = [
    "us-gov-east-1/anthropic.claude-haiku-4-5-20251001-v1:0",
    "us-gov-west-1/anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5-20250929-v1:0",
    "us-gov-east-1/anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us-gov-east-1/claude-sonnet-4-5-20250929-v1:0",
    "us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us-gov-west-1/claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
]


def test_every_model_the_wizard_lists_gets_the_extras_crewai_needs_to_run_it() -> (
    None
):
    from crewai.llm import LLM

    from crewai_cli.create_json_crew import _PROVIDER_MODELS

    listed = [
        f"{provider}/{model_id}"
        for provider, models in _PROVIDER_MODELS.items()
        for model_id, _ in models
    ]
    listed += [f"bedrock/{model_id}" for model_id in _BEDROCK_FEED_MODELS]

    for model in listed:
        routed_to_litellm = LLM._resolve_route(model, {}).native_class is None
        assert ("litellm" in crewai_extras_for_model(model)) is routed_to_litellm, (
            model
        )


def test_classic_provider_extra_covers_every_model_it_lists() -> None:
    """Classic create takes the extra from the provider it lists a model under,
    so a listed model crewAI sends to LiteLLM must sit under a litellm provider."""
    from crewai.llm import LLM

    for provider, models in MODELS.items():
        for model in models:
            if LLM._resolve_route(model, {}).native_class is None:
                assert PROVIDER_EXTRAS.get(provider) == "litellm", model


@pytest.mark.parametrize(
    ("provider", "names"),
    [
        ("openai", ["OPENAI_API_KEY"]),
        ("huggingface", ["HF_TOKEN"]),
        # The deployment name fills MODEL and the API version has a default.
        ("azure", ["AZURE_API_KEY", "AZURE_API_BASE"]),
        ("cerebras", ["CEREBRAS_API_KEY"]),
        ("bedrock", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
        ("watson", ["WATSONX_URL", "WATSONX_APIKEY", "WATSONX_PROJECT_ID"]),
        ("ollama", []),
        ("openrouter", []),
    ],
)
def test_required_key_names_lists_what_a_provider_cannot_run_without(
    provider, names
) -> None:
    assert required_key_names(provider) == names
