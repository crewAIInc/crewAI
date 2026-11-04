from crewai_cli.constants import ENV_VARS, MODELS, PROVIDERS


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


def test_openai_offers_only_the_gpt_6_family() -> None:
    """No GPT-4, GPT-5.x or o-series model: all superseded by GPT-6, and the
    GPT-4.1 snapshots shut down on 2027-01-14."""
    assert MODELS["openai"] == ["gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"]


def test_gemini_models_offer_no_shut_down_model() -> None:
    """`crewai create` must not offer models Google has shut down; picking one
    fails the first run with a ClientError."""
    gemini_models = MODELS["gemini"]
    assert gemini_models[0] == "gemini/gemini-3.8-flash"
    for model in ("gemini/gemini-3.5-flash-lite", "gemini/gemini-4-argon"):
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
    for model in ("gemini-3.5-flash-lite", "gemini-4-argon"):
        assert model in gemini_models
    assert not [m for m in gemini_models if m.startswith(("gemini-2.", "gemini-1."))]


def test_classic_lists_lead_with_current_small_models() -> None:
    """`crewai create crew --classic` lists models in MODELS order."""
    assert MODELS["openai"][0] == "gpt-6-luna"
    assert MODELS["anthropic"][0] == "claude-sonnet-5-5"


def test_wizard_lists_lead_with_current_small_models() -> None:
    """With no key, the wizard shows the curated list in order; the first is the
    `--provider` default."""
    from crewai_cli.create_json_crew import _PROVIDER_MODELS

    assert [model_id for model_id, _ in _PROVIDER_MODELS["openai"]] == [
        "gpt-6-luna",
        "gpt-6.1-sol",
        "gpt-6-astra",
    ]
    assert [model_id for model_id, _ in _PROVIDER_MODELS["anthropic"]] == [
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
    ]


def test_wizard_offers_a_curated_list_for_every_provider() -> None:
    """Without one, the picker shows LiteLLM's feed, which lags and keeps old models."""
    from crewai_cli.create_json_crew import _PROVIDER_MODELS, _PROVIDERS

    assert [key for key, _ in _PROVIDERS if not _PROVIDER_MODELS.get(key)] == []


def test_no_picker_defaults_to_a_preview_model() -> None:
    """A preview can be pulled at short notice, so it is labeled and listed last."""
    from crewai_cli.create_json_crew import _PROVIDER_MODELS

    for provider, models in _PROVIDER_MODELS.items():
        previews = ["(preview)" in label for _, label in models]
        assert previews == sorted(previews), f"{provider}: a preview precedes a GA model"
        assert not previews[0], f"{provider}: the default is a preview"


def _without_provider_prefix(provider: str, model: str) -> str:
    prefix = "watsonx/" if provider == "watson" else f"{provider}/"
    return model.removeprefix(prefix)


def test_classic_and_wizard_offer_the_same_models_in_the_same_order() -> None:
    """Both ways of creating a crew lead with the same default."""
    from crewai_cli.create_json_crew import _PROVIDER_MODELS

    for provider, classic in MODELS.items():
        wizard = [model_id for model_id, _ in _PROVIDER_MODELS[provider]]
        assert [_without_provider_prefix(provider, m) for m in classic] == wizard
