import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.minimax.completion import MiniMaxCompletion


def test_minimax_completion_is_used_when_minimax_provider():
    """MiniMaxCompletion is selected when provider is 'minimax'."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M3", provider="minimax")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.provider == "minimax"
    assert llm.model == "MiniMax-M3"


def test_minimax_completion_is_used_with_prefix():
    """MiniMaxCompletion is selected with 'minimax/' prefix."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="minimax/MiniMax-M3")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.provider == "minimax"
    assert llm.model == "MiniMax-M3"


def test_minimax_inferred_from_model_name():
    """MiniMaxCompletion is inferred from model name without prefix."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M3")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.provider == "minimax"


def test_minimax_m27_model():
    """MiniMax-M2.7 (previous generation) is still recognised."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.7")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.model == "MiniMax-M2.7"


def test_minimax_m27_highspeed_model():
    """MiniMax-M2.7-highspeed is recognised."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M2.7-highspeed")

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.model == "MiniMax-M2.7-highspeed"


def test_minimax_default_model_is_m3():
    """Default model is MiniMax-M3."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = MiniMaxCompletion()

    assert isinstance(llm, MiniMaxCompletion)
    assert llm.model == "MiniMax-M3"


def test_minimax_m3_first_in_model_list():
    """M3 appears first, followed by M2.7 variants; older models are removed."""
    from crewai.llms.constants import MINIMAX_MODELS

    assert MINIMAX_MODELS[0] == "MiniMax-M3"
    assert MINIMAX_MODELS[1] == "MiniMax-M2.7"
    assert MINIMAX_MODELS[2] == "MiniMax-M2.7-highspeed"
    assert "MiniMax-M2.5" not in MINIMAX_MODELS
    assert "MiniMax-M2.5-highspeed" not in MINIMAX_MODELS


def test_minimax_default_base_url():
    """Default base URL points to MiniMax international API."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M3")

    assert llm.base_url == "https://api.minimax.io/v1"


def test_minimax_custom_base_url():
    """Custom base URL can be provided."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(
            model="MiniMax-M3",
            base_url="https://api.minimaxi.com/v1",
        )

    assert llm.base_url == "https://api.minimaxi.com/v1"


def test_minimax_temperature_default():
    """Default temperature is 1.0."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M3")

    assert llm.temperature == 1.0


def test_minimax_temperature_zero_clamped():
    """Temperature of 0 is clamped to 0.01."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M3", temperature=0)

    assert llm.temperature == 0.01


def test_minimax_temperature_valid():
    """Valid temperature in (0, 1] is accepted as-is."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M3", temperature=0.7)

    assert llm.temperature == 0.7


def test_minimax_context_window_size():
    """Context window size matches MiniMax M3 spec (512K)."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        llm = LLM(model="MiniMax-M3")

    # 524288 * 0.85 = 445644
    assert llm.get_context_window_size() == 445644


def test_minimax_requires_api_key():
    """Raises an error when MINIMAX_API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("MINIMAX_API_KEY", None)
        with pytest.raises((ValueError, KeyError, ImportError)):
            LLM(model="MiniMax-M3")
