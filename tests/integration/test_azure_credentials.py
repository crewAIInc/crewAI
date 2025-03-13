import pytest

from src.crewai.llm import LLM


def test_azure_detection_with_credentials():
    """Test that Azure is detected correctly when credentials are provided but model lacks azure/ prefix."""
    # Create LLM instance with Azure parameters but without azure/ prefix
    llm = LLM(
        api_key='test_key',
        api_base='test_base',
        model='gpt-4o-mini-2024-07-18',  # Model from issue #2358
        api_version='test_version'
    )
    
    # Check if provider is detected correctly
    provider = llm._get_custom_llm_provider()
    assert provider == 'azure', "Azure provider should be detected based on credentials"
    
    # Prepare parameters that would be passed to LiteLLM
    params = llm._prepare_completion_params(messages=[{"role": "user", "content": "test"}])
    assert params.get('api_key') == 'test_key', "API key should be included in params"
    assert params.get('api_base') == 'test_base', "API base should be included in params"
    assert params.get('api_version') == 'test_version', "API version should be included in params"


def test_azure_validation_error():
    """Test that validation error is raised when Azure credentials are incomplete."""
    # Create LLM instance with incomplete Azure parameters
    llm = LLM(
        model='azure/gpt-4',
        api_key='test_key',
        # Missing api_base and api_version
    )
    
    # Validation should fail
    with pytest.raises(ValueError) as excinfo:
        llm._validate_call_params()
    
    assert "Incomplete Azure credentials" in str(excinfo.value)
