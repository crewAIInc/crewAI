from crewai.cli.constants import ENV_VARS, MODELS, PROVIDERS


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


def test_openai_models_include_latest():
    """Test that OpenAI models include the latest models."""
    assert "openai" in MODELS
    openai_models = MODELS["openai"]
    assert len(openai_models) > 0
    assert "gpt-4o" in openai_models
    assert "gpt-4o-mini" in openai_models
    assert "o1" in openai_models
    assert "o3" in openai_models
    assert "o3-mini" in openai_models


def test_anthropic_models_include_latest():
    """Test that Anthropic models include the latest Claude models."""
    assert "anthropic" in MODELS
    anthropic_models = MODELS["anthropic"]
    assert len(anthropic_models) > 0
    assert "claude-3-7-sonnet-20250219" in anthropic_models
    assert "claude-3-5-sonnet-20241022" in anthropic_models
    assert "claude-3-5-haiku-20241022" in anthropic_models


def test_groq_models_include_latest():
    """Test that Groq models include the latest Llama models."""
    assert "groq" in MODELS
    groq_models = MODELS["groq"]
    assert len(groq_models) > 0
    assert "groq/llama-3.3-70b-versatile" in groq_models


def test_ollama_models_include_latest():
    """Test that Ollama models include the latest models."""
    assert "ollama" in MODELS
    ollama_models = MODELS["ollama"]
    assert len(ollama_models) > 0
    assert "ollama/llama3.2" in ollama_models
    assert "ollama/llama3.3" in ollama_models


def test_all_providers_have_models():
    """Test that all providers in PROVIDERS have corresponding models in MODELS."""
    providers_with_models = [
        "openai",
        "anthropic",
        "gemini",
        "nvidia_nim",
        "groq",
        "ollama",
        "watson",
        "bedrock",
        "huggingface",
        "sambanova",
    ]
    for provider in providers_with_models:
        assert provider in MODELS, f"Provider {provider} should have models defined"
        assert len(MODELS[provider]) > 0, f"Provider {provider} should have at least one model"


def test_all_providers_have_env_vars_or_defaults():
    """Test that all providers have environment variable configurations."""
    for provider in PROVIDERS:
        if provider in ENV_VARS:
            assert len(ENV_VARS[provider]) > 0, f"Provider {provider} should have env var config"
