import os
import sys
import types
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.avian.completion import AvianCompletion


def test_avian_completion_is_used_when_avian_provider():
    """Test that AvianCompletion is used when model has 'avian/' prefix."""
    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        llm = LLM(model="avian/deepseek/deepseek-v3.2")

    assert llm.__class__.__name__ == "AvianCompletion"
    assert llm.provider == "avian"
    assert llm.model == "deepseek/deepseek-v3.2"


def test_avian_completion_isinstance():
    """Test that the LLM instance is an AvianCompletion."""
    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        llm = LLM(model="avian/deepseek/deepseek-v3.2")

    assert isinstance(llm, AvianCompletion)


def test_avian_completion_sets_base_url():
    """Test that AvianCompletion sets the correct base URL."""
    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        llm = LLM(model="avian/deepseek/deepseek-v3.2")

    assert llm.base_url == "https://api.avian.io/v1"


def test_avian_completion_uses_env_api_key():
    """Test that AvianCompletion reads AVIAN_API_KEY from environment."""
    with patch.dict(os.environ, {"AVIAN_API_KEY": "sk-avian-test-123"}):
        llm = LLM(model="avian/deepseek/deepseek-v3.2")

    assert llm.api_key == "sk-avian-test-123"


def test_avian_completion_uses_explicit_api_key():
    """Test that AvianCompletion accepts an explicit api_key parameter."""
    with patch.dict(os.environ, {}, clear=False):
        # Remove AVIAN_API_KEY if present
        env = {k: v for k, v in os.environ.items() if k != "AVIAN_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            llm = LLM(
                model="avian/deepseek/deepseek-v3.2",
                api_key="sk-explicit-key",
            )

    assert llm.api_key == "sk-explicit-key"


def test_avian_completion_raises_without_api_key():
    """Test that AvianCompletion raises ValueError when no API key is available."""
    env = {k: v for k, v in os.environ.items() if k != "AVIAN_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises((ValueError, ImportError)):
            LLM(model="avian/deepseek/deepseek-v3.2")


def test_avian_completion_custom_base_url():
    """Test that AvianCompletion accepts a custom base_url."""
    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        llm = LLM(
            model="avian/deepseek/deepseek-v3.2",
            base_url="https://custom.avian.io/v1",
        )

    assert llm.base_url == "https://custom.avian.io/v1"


def test_avian_completion_all_models():
    """Test that all Avian models can be instantiated."""
    models = [
        "avian/deepseek/deepseek-v3.2",
        "avian/moonshotai/kimi-k2.5",
        "avian/z-ai/glm-5",
        "avian/minimax/minimax-m2.5",
    ]

    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        for model in models:
            llm = LLM(model=model)
            assert isinstance(llm, AvianCompletion), f"Failed for model: {model}"
            assert llm.provider == "avian"


def test_avian_completion_module_is_imported():
    """Test that the completion module is properly imported when using Avian provider."""
    module_name = "crewai.llms.providers.avian.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        LLM(model="avian/deepseek/deepseek-v3.2")

    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)
    assert hasattr(completion_mod, "AvianCompletion")


def test_avian_with_explicit_provider_kwarg():
    """Test that explicit provider='avian' kwarg routes to AvianCompletion."""
    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        llm = LLM(
            model="deepseek/deepseek-v3.2",
            provider="avian",
        )

    assert llm.__class__.__name__ == "AvianCompletion"
    assert llm.provider == "avian"
    assert llm.model == "deepseek/deepseek-v3.2"


def test_avian_validate_model_in_constants():
    """Test that Avian models are validated correctly in constants."""
    assert LLM._validate_model_in_constants("deepseek/deepseek-v3.2", "avian") is True
    assert LLM._validate_model_in_constants("moonshotai/kimi-k2.5", "avian") is True
    assert LLM._validate_model_in_constants("z-ai/glm-5", "avian") is True
    assert LLM._validate_model_in_constants("minimax/minimax-m2.5", "avian") is True
    assert LLM._validate_model_in_constants("unknown-model", "avian") is False


def test_avian_completion_with_temperature():
    """Test that AvianCompletion passes through temperature parameter."""
    with patch.dict(os.environ, {"AVIAN_API_KEY": "test-key"}):
        llm = LLM(model="avian/deepseek/deepseek-v3.2", temperature=0.7)

    assert llm.temperature == 0.7


def test_avian_env_base_url_override():
    """Test that AVIAN_API_BASE env var overrides the default base URL."""
    with patch.dict(
        os.environ,
        {"AVIAN_API_KEY": "test-key", "AVIAN_API_BASE": "https://staging.avian.io/v1"},
    ):
        llm = LLM(model="avian/deepseek/deepseek-v3.2")

    assert llm.base_url == "https://staging.avian.io/v1"
