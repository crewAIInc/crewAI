from unittest.mock import patch

from crewai.cli.constants import MODELS
from crewai.cli.provider import select_model


def test_watsonx_models_include_llama4_maverick():
    """Test that the watsonx models list includes the Llama 4 Maverick model."""
    watsonx_models = MODELS.get("watson", [])
    assert "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8" in watsonx_models


def test_select_model_watsonx_llama4_maverick():
    """Test that the Llama 4 Maverick model can be selected for watsonx provider."""
    provider = "watson"
    provider_models = {}
    
    with patch("crewai.cli.provider.select_choice") as mock_select_choice:
        mock_select_choice.return_value = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        
        result = select_model(provider, provider_models)
        
        assert result == "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        mock_select_choice.assert_called_once()
        
        call_args = mock_select_choice.call_args
        available_models = call_args[0][1]
        assert "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8" in available_models


def test_watsonx_model_list_ordering():
    """Test that watsonx models are properly ordered."""
    watsonx_models = MODELS.get("watson", [])
    
    expected_models = [
        "watsonx/meta-llama/llama-3-1-70b-instruct",
        "watsonx/meta-llama/llama-3-1-8b-instruct",
        "watsonx/meta-llama/llama-3-2-11b-vision-instruct",
        "watsonx/meta-llama/llama-3-2-1b-instruct",
        "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
        "watsonx/meta-llama/llama-3-405b-instruct",
        "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "watsonx/mistral/mistral-large",
        "watsonx/ibm/granite-3-8b-instruct",
    ]
    
    assert watsonx_models == expected_models
