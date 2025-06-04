"""Tests for AI/ML API integration with CrewAI."""

from unittest.mock import patch

from crewai.llm import LLM
from crewai.utilities.llm_utils import create_llm


class TestAIMLAPIIntegration:
    """Test suite for AI/ML API provider integration."""

    def test_aiml_api_model_context_windows(self):
        """Test that AI/ML API models have correct context window sizes."""
        test_cases = [
            ("openai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", 131072),
            ("openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", 131072),
            ("openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 131072),
            ("openai/anthropic/claude-3-5-sonnet-20241022", 200000),
            ("openai/anthropic/claude-3-5-haiku-20241022", 200000),
            ("openai/mistralai/Mistral-7B-Instruct-v0.3", 32768),
            ("openai/Qwen/Qwen2.5-72B-Instruct-Turbo", 131072),
            ("openai/deepseek-ai/DeepSeek-V2.5", 131072),
        ]
        
        for model_name, expected_context_size in test_cases:
            llm = LLM(model=model_name)
            expected_usable_size = int(expected_context_size * 0.85)
            actual_context_size = llm.get_context_window_size()
            assert actual_context_size == expected_usable_size, (
                f"Model {model_name} should have context window size {expected_usable_size}, "
                f"but got {actual_context_size}"
            )

    def test_aiml_api_provider_detection(self):
        """Test that AI/ML API models are correctly identified as openai provider."""
        llm = LLM(model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        provider = llm._get_custom_llm_provider()
        assert provider == "openai", f"Expected provider 'openai', but got '{provider}'"

    def test_aiml_api_model_instantiation(self):
        """Test that AI/ML API models can be instantiated correctly."""
        model_names = [
            "openai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "openai/anthropic/claude-3-5-sonnet-20241022",
            "openai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            "openai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        ]
        
        for model_name in model_names:
            llm = LLM(model=model_name)
            assert llm.model == model_name
            assert llm._get_custom_llm_provider() == "openai"
            assert llm.get_context_window_size() > 0

    @patch('crewai.llm.litellm.utils.supports_function_calling')
    def test_aiml_api_function_calling_support(self, mock_supports_function_calling):
        """Test function calling support detection for AI/ML API models."""
        mock_supports_function_calling.return_value = True
        
        llm = LLM(model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        supports_fc = llm.supports_function_calling()
        
        assert supports_fc is True
        mock_supports_function_calling.assert_called_once_with(
            "openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            custom_llm_provider="openai"
        )

    def test_aiml_api_with_create_llm(self):
        """Test that AI/ML API models work with create_llm utility."""
        model_name = "openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        llm = create_llm(model_name)
        
        assert isinstance(llm, LLM)
        assert llm.model == model_name
        assert llm._get_custom_llm_provider() == "openai"

    def test_aiml_api_model_validation(self):
        """Test that AI/ML API models pass validation checks."""
        llm = LLM(model="openai/anthropic/claude-3-5-sonnet-20241022")
        
        llm._validate_call_params()
        
        llm_with_format = LLM(
            model="openai/anthropic/claude-3-5-sonnet-20241022",
            response_format={"type": "json_object"}
        )
        try:
            llm_with_format._validate_call_params()
        except ValueError as e:
            assert "does not support response_format" in str(e)

    def test_aiml_api_context_window_bounds(self):
        """Test that AI/ML API model context windows are within valid bounds."""
        from crewai.llm import LLM_CONTEXT_WINDOW_SIZES
        
        aiml_models = {k: v for k, v in LLM_CONTEXT_WINDOW_SIZES.items() 
                      if k.startswith("openai/")}
        
        MIN_CONTEXT = 1024
        MAX_CONTEXT = 2097152
        
        for model_name, context_size in aiml_models.items():
            assert MIN_CONTEXT <= context_size <= MAX_CONTEXT, (
                f"Model {model_name} context window {context_size} is outside "
                f"valid bounds [{MIN_CONTEXT}, {MAX_CONTEXT}]"
            )

    def test_aiml_api_model_prefixes(self):
        """Test that all AI/ML API models use the correct openai/ prefix."""
        from crewai.llm import LLM_CONTEXT_WINDOW_SIZES
        
        aiml_models = [k for k in LLM_CONTEXT_WINDOW_SIZES.keys() 
                      if k.startswith("openai/")]
        
        assert len(aiml_models) > 0, "No AI/ML API models found in context window sizes"
        
        for model_name in aiml_models:
            assert model_name.startswith("openai/"), (
                f"AI/ML API model {model_name} should start with 'openai/' prefix"
            )
            parts = model_name.split("/")
            assert len(parts) >= 3, (
                f"AI/ML API model {model_name} should have format 'openai/provider/model'"
            )


class TestAIMLAPIExamples:
    """Test examples of using AI/ML API with CrewAI components."""

    def test_aiml_api_with_agent_example(self):
        """Test example usage of AI/ML API with CrewAI Agent."""
        from crewai import Agent
        
        llm = LLM(model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        
        agent = Agent(
            role="AI Assistant",
            goal="Help users with their questions",
            backstory="You are a helpful AI assistant powered by Llama 3.1",
            llm=llm,
        )
        
        assert agent.llm.model == "openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        assert agent.llm._get_custom_llm_provider() == "openai"

    def test_aiml_api_different_model_types(self):
        """Test different types of models available through AI/ML API."""
        model_types = {
            "llama": "openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "claude": "openai/anthropic/claude-3-5-sonnet-20241022",
            "mistral": "openai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            "qwen": "openai/Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek": "openai/deepseek-ai/DeepSeek-V2.5",
        }
        
        for model_type, model_name in model_types.items():
            llm = LLM(model=model_name)
            assert llm.model == model_name
            assert llm._get_custom_llm_provider() == "openai"
            assert llm.get_context_window_size() > 0, (
                f"{model_type} model should have positive context window"
            )
