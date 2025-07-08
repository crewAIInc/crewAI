"""Tests for DeepSeek integration in CrewAI."""

from unittest.mock import patch

from crewai.llm import LLM
from crewai.cli.constants import ENV_VARS, PROVIDERS, MODELS


class TestDeepSeekIntegration:
    """Test DeepSeek integration in CrewAI."""

    def test_deepseek_in_providers(self):
        """Test that DeepSeek is included in the providers list."""
        assert "deepseek" in PROVIDERS

    def test_deepseek_in_env_vars(self):
        """Test that DeepSeek API key configuration is in ENV_VARS."""
        assert "deepseek" in ENV_VARS
        deepseek_config = ENV_VARS["deepseek"]
        assert len(deepseek_config) == 1
        assert deepseek_config[0]["key_name"] == "DEEPSEEK_API_KEY"
        assert "DeepSeek API key" in deepseek_config[0]["prompt"]

    def test_deepseek_in_models(self):
        """Test that DeepSeek models are included in the models dictionary."""
        assert "deepseek" in MODELS
        deepseek_models = MODELS["deepseek"]
        expected_models = [
            "deepseek/deepseek-chat",
            "deepseek/deepseek-coder",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-v3",
            "deepseek/deepseek-reasoner",
        ]
        for model in expected_models:
            assert model in deepseek_models

    def test_llm_creation_with_deepseek_chat(self):
        """Test creating LLM instance with deepseek-chat model."""
        llm = LLM(model="deepseek-chat")
        assert llm.model == "deepseek-chat"
        assert llm.get_context_window_size() > 0

    def test_llm_creation_with_deepseek_prefix(self):
        """Test creating LLM instance with deepseek/ prefix."""
        llm = LLM(model="deepseek/deepseek-chat")
        assert llm.model == "deepseek/deepseek-chat"
        assert llm._get_custom_llm_provider() == "deepseek"
        assert llm.get_context_window_size() > 0

    def test_deepseek_context_window_sizes(self):
        """Test that all DeepSeek models have context window sizes defined."""
        from crewai.llm import LLM_CONTEXT_WINDOW_SIZES
        
        deepseek_models = [
            "deepseek-chat",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-coder",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-v3",
            "deepseek/deepseek-reasoner",
        ]
        
        for model in deepseek_models:
            assert model in LLM_CONTEXT_WINDOW_SIZES
            assert LLM_CONTEXT_WINDOW_SIZES[model] > 0

    def test_deepseek_models_context_window_consistency(self):
        """Test that DeepSeek models have consistent context window sizes."""
        from crewai.llm import LLM_CONTEXT_WINDOW_SIZES
        
        expected_size = 128000
        deepseek_models = [
            "deepseek-chat",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-coder",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-v3",
            "deepseek/deepseek-reasoner",
        ]
        
        for model in deepseek_models:
            assert LLM_CONTEXT_WINDOW_SIZES[model] == expected_size

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"})
    def test_llm_with_deepseek_api_key(self):
        """Test LLM creation with DeepSeek API key in environment."""
        llm = LLM(model="deepseek/deepseek-chat")
        assert llm.model == "deepseek/deepseek-chat"
        assert llm._get_custom_llm_provider() == "deepseek"

    def test_deepseek_provider_detection(self):
        """Test that DeepSeek provider is correctly detected from model name."""
        llm = LLM(model="deepseek/deepseek-chat")
        provider = llm._get_custom_llm_provider()
        assert provider == "deepseek"

    def test_deepseek_vs_openrouter_provider_detection(self):
        """Test provider detection for DeepSeek vs OpenRouter DeepSeek models."""
        deepseek_llm = LLM(model="deepseek/deepseek-chat")
        openrouter_llm = LLM(model="openrouter/deepseek/deepseek-chat")
        
        assert deepseek_llm._get_custom_llm_provider() == "deepseek"
        assert openrouter_llm._get_custom_llm_provider() == "openrouter"

    def test_all_deepseek_models_can_be_instantiated(self):
        """Test that all DeepSeek models in MODELS can be instantiated."""
        deepseek_models = MODELS["deepseek"]
        
        for model in deepseek_models:
            llm = LLM(model=model)
            assert llm.model == model
            assert llm._get_custom_llm_provider() == "deepseek"
            assert llm.get_context_window_size() > 0
