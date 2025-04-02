import pytest

from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM, PRO_CONTEXT_SIZE


def test_get_custom_llm_provider_gemini_2_5():
    """Test that the Gemini 2.5 model is correctly identified as a Gemini provider."""
    llm = LLM(model="gemini/gemini-2.5-pro-exp-03-25")
    assert llm._get_custom_llm_provider() == "gemini"

def test_gemini_2_5_context_window_size():
    """Test that the Gemini 2.5 model has the correct context window size."""
    llm = LLM(model="gemini-2.5-pro-exp-03-25")
    expected_size = int(PRO_CONTEXT_SIZE * CONTEXT_WINDOW_USAGE_RATIO)
    assert llm.get_context_window_size() == expected_size

def test_gemini_2_5_invalid_model_name():
    """Test handling of invalid model name variations."""
    llm = LLM(model="gemini-2.5-wrong")
    assert llm._get_custom_llm_provider() != "gemini"

def test_gemini_2_5_model_parameters():
    """Test model initialization with various parameters."""
    llm = LLM(
        model="gemini/gemini-2.5-pro-exp-03-25", 
        temperature=0.7, 
        max_tokens=1000
    )
    assert llm.model == "gemini/gemini-2.5-pro-exp-03-25"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 1000

def test_gemini_2_5_with_and_without_prefix():
    """Test that the model works with and without the 'gemini/' prefix."""
    llm_with_prefix = LLM(model="gemini/gemini-2.5-pro-exp-03-25")
    llm_without_prefix = LLM(model="gemini-2.5-pro-exp-03-25")
    
    assert llm_with_prefix._get_custom_llm_provider() == "gemini"
    assert llm_without_prefix._get_custom_llm_provider() == "gemini"
    
    assert llm_with_prefix.get_context_window_size() == llm_without_prefix.get_context_window_size()
