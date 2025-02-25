import pytest

from crewai.llm import LLM


def test_numeric_model_id_validation():
    # Test with integer model ID
    with pytest.raises(ValueError, match="Invalid model ID: 3420. Model ID cannot be a numeric value without a provider prefix."):
        LLM(model=3420)
    
    # Test with string numeric model ID
    with pytest.raises(ValueError, match="Invalid model ID: 3420. Model ID cannot be a numeric value without a provider prefix."):
        LLM(model="3420")
    
    # Test with valid model ID
    llm = LLM(model="openai/gpt-4")
    assert llm.model == "openai/gpt-4"
    
    # Test with valid model ID that contains numbers
    llm = LLM(model="gpt-3.5-turbo")
    assert llm.model == "gpt-3.5-turbo"
