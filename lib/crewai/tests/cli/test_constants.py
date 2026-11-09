from crewai.constants import ENV_VARS, MODELS, PROVIDERS


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


def test_the_cli_asks_for_the_key_of_the_model_crewai_runs_by_default():
    """The CLI keeps its own copy of the default, to pick the API key to ask for."""
    from crewai.constants import DEFAULT_LLM_MODEL
    from crewai_cli.constants import DEFAULT_LLM_MODEL as CLI_DEFAULT_LLM_MODEL

    assert CLI_DEFAULT_LLM_MODEL == DEFAULT_LLM_MODEL
