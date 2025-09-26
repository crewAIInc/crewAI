"""Tests for Watson provider environment variable handling."""

import os
from unittest.mock import patch
import pytest
from crewai.rag.embeddings.providers.ibm.watson import WatsonProvider


class TestWatsonEnvironmentVariables:
    """Test Watson provider environment variable compatibility."""

    def test_watsonx_prefix_variables(self):
        """Test that WATSONX_ prefixed variables work correctly."""
        with patch.dict(os.environ, {
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSONX_APIKEY": "test-api-key",
            "WATSONX_PROJECT_ID": "test-project-id",
            "WATSONX_MODEL_ID": "ibm/slate-125m-english-rtrvr"
        }, clear=True):
            provider = WatsonProvider()
            assert provider.url == "https://us-south.ml.cloud.ibm.com"
            assert provider.api_key == "test-api-key"
            assert provider.project_id == "test-project-id"
            assert provider.model_id == "ibm/slate-125m-english-rtrvr"

    def test_watson_prefix_backward_compatibility(self):
        """Test that legacy WATSON_ prefixed variables still work."""
        with patch.dict(os.environ, {
            "WATSON_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSON_API_KEY": "test-api-key",
            "WATSON_PROJECT_ID": "test-project-id",
            "WATSON_MODEL_ID": "ibm/slate-125m-english-rtrvr"
        }, clear=True):
            provider = WatsonProvider()
            assert provider.url == "https://us-south.ml.cloud.ibm.com"
            assert provider.api_key == "test-api-key"
            assert provider.project_id == "test-project-id"
            assert provider.model_id == "ibm/slate-125m-english-rtrvr"

    def test_watsonx_takes_precedence_over_watson(self):
        """Test that WATSONX_ variables take precedence over WATSON_ when both are set."""
        with patch.dict(os.environ, {
            "WATSONX_URL": "https://new-url.com",
            "WATSON_URL": "https://old-url.com",
            "WATSONX_APIKEY": "new-key",
            "WATSON_API_KEY": "old-key",
            "WATSONX_PROJECT_ID": "new-project",
            "WATSON_PROJECT_ID": "old-project",
            "WATSONX_MODEL_ID": "new-model",
            "WATSON_MODEL_ID": "old-model"
        }, clear=True):
            provider = WatsonProvider()
            assert provider.url == "https://new-url.com"
            assert provider.api_key == "new-key"
            assert provider.project_id == "new-project"
            assert provider.model_id == "new-model"

    def test_mixed_environment_variables(self):
        """Test that mixing WATSONX_ and WATSON_ variables works correctly."""
        with patch.dict(os.environ, {
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSON_API_KEY": "test-api-key",
            "WATSONX_PROJECT_ID": "test-project-id",
            "WATSON_MODEL_ID": "ibm/slate-125m-english-rtrvr"
        }, clear=True):
            provider = WatsonProvider()
            assert provider.url == "https://us-south.ml.cloud.ibm.com"
            assert provider.api_key == "test-api-key"
            assert provider.project_id == "test-project-id"
            assert provider.model_id == "ibm/slate-125m-english-rtrvr"

    def test_token_environment_variables(self):
        """Test that token environment variables work with both prefixes."""
        with patch.dict(os.environ, {
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSONX_APIKEY": "test-api-key",
            "WATSONX_PROJECT_ID": "test-project-id",
            "WATSONX_MODEL_ID": "ibm/slate-125m-english-rtrvr",
            "WATSONX_TOKEN": "test-token"
        }, clear=True):
            provider = WatsonProvider()
            assert provider.token == "test-token"

        with patch.dict(os.environ, {
            "WATSON_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSON_API_KEY": "test-api-key",
            "WATSON_PROJECT_ID": "test-project-id",
            "WATSON_MODEL_ID": "ibm/slate-125m-english-rtrvr",
            "WATSON_TOKEN": "legacy-token"
        }, clear=True):
            provider = WatsonProvider()
            assert provider.token == "legacy-token"

    def test_validation_error_when_required_fields_missing(self):
        """Test that validation errors are raised when required fields are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):
                WatsonProvider()

    def test_space_or_project_validation(self):
        """Test that either space_id or project_id must be provided."""
        with patch.dict(os.environ, {
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSONX_APIKEY": "test-api-key",
            "WATSONX_MODEL_ID": "ibm/slate-125m-english-rtrvr"
        }, clear=True):
            with pytest.raises(ValueError, match="One of 'space_id' or 'project_id' must be provided"):
                WatsonProvider()
