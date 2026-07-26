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


def test_atlascloud_provider_configuration():
    """Test that Atlas Cloud is available with credentials and model presets."""
    assert "atlascloud" in PROVIDERS
    assert any(
        detail.get("key_name") == "ATLASCLOUD_API_KEY"
        for detail in ENV_VARS["atlascloud"]
    )
    assert MODELS["atlascloud"][0] == ("atlascloud/deepseek-ai/deepseek-v4-pro")
