import pytest

def test_import_llm_from_crewai():
    """Test that LLM can be imported directly from crewai."""
    try:
        from crewai import LLM
        assert LLM is not None
    except ImportError as e:
        pytest.fail(f"Failed to import LLM from crewai: {e}")

def test_bedrock_llm_creation():
    """Test that a Bedrock LLM can be created."""
    try:
        from crewai import LLM
        
        # Just test the object creation, not the actual API call
        bedrock_llm = LLM(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        )
        assert bedrock_llm is not None
        assert bedrock_llm.model == "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    except Exception as e:
        pytest.fail(f"Failed to create Bedrock LLM: {e}")
