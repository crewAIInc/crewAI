import pytest
from unittest.mock import patch

from crewai.llm import LLM


def test_empty_messages_validation():
    """
    Test that LLM.call() raises a ValueError when an empty messages list is passed.
    This prevents the IndexError in LiteLLM's ollama_pt() function.
    """
    llm = LLM(model="gpt-3.5-turbo")  # Any model will do for this test
    
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.call(messages=[])
    
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.call(messages=None)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ollama_model_empty_messages():
    """
    Test that LLM.call() with an Ollama model raises a ValueError 
    when an empty messages list is passed.
    """
    llm = LLM(model="ollama/llama3")
    
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.call(messages=[])
