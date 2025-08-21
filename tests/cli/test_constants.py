from crewai.cli.constants import ENV_VARS, MODELS, PROVIDERS


def test_huggingface_in_providers():
    """Test that Huggingface is in the PROVIDERS list."""
    assert "huggingface" in PROVIDERS


def test_huggingface_env_vars():
    """Test that Huggingface environment variables are properly configured."""
    assert "huggingface" in ENV_VARS
    assert any(
        detail.get("key_name") == "HF_TOKEN"
        for detail in ENV_VARS["huggingface"]
    )


def test_huggingface_models():
    """Test that Huggingface models are properly configured."""
    assert "huggingface" in MODELS
    assert len(MODELS["huggingface"]) > 0


def test_openai_models_include_latest():
    """Test that OpenAI models include the latest GPT-5 series."""
    openai_models = MODELS["openai"]
    assert "gpt-5" in openai_models
    assert "gpt-5-mini" in openai_models
    assert "gpt-5-nano" in openai_models
    assert "gpt-4.1" in openai_models
    assert "o3-mini" in openai_models


def test_anthropic_models_include_latest():
    """Test that Anthropic models include the latest Claude 4 series."""
    anthropic_models = MODELS["anthropic"]
    assert "claude-3.7-sonnet-20250219" in anthropic_models
    assert "claude-4-sonnet-20250301" in anthropic_models
    assert "claude-4.1-opus-20250315" in anthropic_models


def test_gemini_models_include_latest():
    """Test that Gemini models include the latest 2.5 series."""
    gemini_models = MODELS["gemini"]
    assert "gemini/gemini-2.5-pro" in gemini_models
    assert "gemini/gemini-2.5-flash" in gemini_models
    assert "gemini/gemini-2.5-flash-lite" in gemini_models


def test_all_providers_have_models():
    """Test that all providers in PROVIDERS have corresponding models."""
    for provider in PROVIDERS:
        if provider in MODELS:
            assert len(MODELS[provider]) > 0, f"Provider {provider} has no models"


def test_model_format_consistency():
    """Test that model names follow consistent formatting patterns."""
    for provider, models in MODELS.items():
        for model in models:
            assert isinstance(model, str), f"Model {model} in {provider} is not a string"
            assert len(model.strip()) > 0, f"Empty model name in {provider}"
