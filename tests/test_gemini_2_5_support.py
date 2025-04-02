import pytest
from crewai.llm import LLM

def test_get_custom_llm_provider_gemini_2_5():
    """Test that the Gemini 2.5 model is correctly identified as a Gemini provider."""
    llm = LLM(model="gemini/gemini-2.5-pro-exp-03-25")
    assert llm._get_custom_llm_provider() == "gemini"

def test_gemini_2_5_context_window_size():
    """Test that the Gemini 2.5 model has the correct context window size."""
    llm = LLM(model="gemini-2.5-pro-exp-03-25")
    from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO
    expected_size = int(2097152 * CONTEXT_WINDOW_USAGE_RATIO)
    assert llm.get_context_window_size() == expected_size
