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
    for model in ("gemini/gemini-4-argon", "gemini/gemini-3.6-flash"):
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
