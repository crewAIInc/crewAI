
import os
import pytest
from unittest.mock import patch
from crewai.llm import LLM
from crewai.llms.providers.azure.completion import AzureCompletion

def test_custom_azure_endpoint_construction():
    """
    Test that custom Azure endpoints (e.g. cservices.azure.com) are correctly handled.
    Regression test for issue #4260.
    """
    model = "azure/gpt-4"
    # A custom Azure domain
    custom_base_endpoint = "https://my-custom-gateway.cservices.azure.com"
    expected_full_endpoint = "https://my-custom-gateway.cservices.azure.com/openai/deployments/gpt-4"
    
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "fake-key",
        "AZURE_ENDPOINT": custom_base_endpoint
    }):
        llm = LLM(model=model)
        
        # Verify it is using AzureCompletion
        assert isinstance(llm, AzureCompletion)
        
        # Verify the endpoint was correctly expanded
        assert llm.endpoint == expected_full_endpoint
        assert llm.is_azure_openai_endpoint == True

def test_standard_azure_endpoint_construction():
    """Verify standard openai.azure.com still works correctly"""
    model = "azure/gpt-4"
    standard_endpoint = "https://my-resource.openai.azure.com"
    expected_endpoint = "https://my-resource.openai.azure.com/openai/deployments/gpt-4"
    
    with patch.dict(os.environ, {
        "AZURE_API_KEY": "fake-key",
        "AZURE_ENDPOINT": standard_endpoint
    }):
        llm = LLM(model=model)
        assert llm.endpoint == expected_endpoint
        assert llm.is_azure_openai_endpoint == True
