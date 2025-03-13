"""Tests for LLM import functionality."""

import pytest


@pytest.fixture
def bedrock_model_path():
    """Fixture providing the standard Bedrock model path for testing."""
    return "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"


def test_import_llm_from_crewai():
    """
    Test LLM import functionality from crewai package.
    
    Verifies:
        - Direct import of LLM class from crewai package
        - Import statement succeeds without raising exceptions
    """
    try:
        from crewai import LLM
        assert LLM is not None
    except ImportError as e:
        pytest.fail(f"Failed to import LLM from crewai: {e}")


def test_bedrock_llm_creation(bedrock_model_path):
    """
    Test that a Bedrock LLM can be created with the correct model.
    
    Verifies:
        - LLM can be instantiated with a Bedrock model
        - The model property is correctly set to the Bedrock model path
        - No exceptions are raised during instantiation
    """
    try:
        from crewai import LLM
        
        # Just test the object creation, not the actual API call
        bedrock_llm = LLM(model=bedrock_model_path)
        assert bedrock_llm is not None
        assert bedrock_llm.model == bedrock_model_path
    except Exception as e:
        pytest.fail(f"Failed to create Bedrock LLM: {e}")



def test_llm_with_invalid_model():
    """
    Test LLM creation with an invalid model name format.
    
    Verifies:
        - LLM can be instantiated with any model string
        - No validation errors occur during instantiation
        - The model property is correctly set to the provided string
    """
    try:
        from crewai import LLM
        
        invalid_model = "invalid-model-name"
        llm = LLM(model=invalid_model)
        assert llm is not None
        assert llm.model == invalid_model
    except Exception as e:
        pytest.fail(f"Failed to create LLM with invalid model: {e}")
