import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.minimax.completion import MiniMaxCompletion


def test_minimax_completion_is_used_when_minimax_provider():
    """MiniMaxCompletion is selected when provider is 'minimax'."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5", provider="minimax")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.provider == "minimax"
    assert llm.model == "MiniMax-M2.5"


def test_minimax_completion_is_used_with_prefix():
    """MiniMaxCompletion is selected with 'minimax/' prefix."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="minimax/MiniMax-M2.5")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.provider == "minimax"
    assert llm.model == "MiniMax-M2.5"


def test_minimax_inferred_from_model_name():
    """MiniMaxCompletion is inferred from model name without prefix."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.provider == "minimax"


def test_minimax_highspeed_model():
    """MiniMax-M2.5-highspeed is recognised."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5-highspeed")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.model == "MiniMax-M2.5-highspeed"


def test_minimax_default_base_url():
    """Default base URL points to MiniMax international API."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5")

    assert llm.base_url == "https://api.minimax.io/v1"


def test_minimax_custom_base_url():
    """Custom base URL can be provided."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(
            model="MiniMax-M2.5",
            base_url="https://api.minimaxi.com/v1",
        )

    assert llm.base_url == "https://api.minimaxi.com/v1"


def test_minimax_temperature_default():
    """Default temperature is 1.0."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5")

    assert llm.temperature == 1.0


def test_minimax_temperature_zero_clamped():
    """Temperature of 0 is clamped to 0.01."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5", temperature=0)

    assert llm.temperature == 0.01


def test_minimax_temperature_valid():
    """Valid temperature in (0, 1] is accepted as-is."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5", temperature=0.7)

    assert llm.temperature == 0.7


def test_minimax_context_window_size():
    """Context window size matches MiniMax spec."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.5")

    # 204800 * 0.85 = 174080
    assert llm.get_context_window_size() == 174080


def test_minimax_requires_api_key():
    """Raises an error when MINIMAX_API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("MINIMAX_API_KEY", None)
        with pytest.raises((ValueError, KeyError, ImportError)):
            LLM(model="MiniMax-M2.5")
