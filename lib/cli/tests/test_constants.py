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
