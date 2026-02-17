"""Tests for platform tools misc functionality."""

import os
from unittest.mock import patch

import pytest
from crewai.context import platform_integration_context
from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_integration_token,
)



class TestTokenRetrievalWithFallback:
    """Test token retrieval logic with environment fallback."""

    def test_context_token_takes_precedence(self, clean_context):
        """Test that context token takes precedence over environment variable."""
        context_token = "context-token"
        env_token = "env-token"

        with patch.dict(os.environ, {"CREWAI_PLATFORM_INTEGRATION_TOKEN": env_token}):
            with platform_integration_context(context_token):
                token = get_platform_integration_token()
                assert token == context_token

    def test_environment_fallback_when_no_context(self, clean_context):
        """Test fallback to environment variable when no context token."""
        env_token = "env-fallback-token"

        with patch.dict(os.environ, {"CREWAI_PLATFORM_INTEGRATION_TOKEN": env_token}):
            token = get_platform_integration_token()
            assert token == env_token

    @pytest.mark.parametrize("empty_value", ["", None])
    def test_missing_token_raises_error(self, clean_context, empty_value):
        """Test that missing tokens raise appropriate errors."""
        env_dict = {"CREWAI_PLATFORM_INTEGRATION_TOKEN": empty_value} if empty_value is not None else {}

        with patch.dict(os.environ, env_dict, clear=True):
            with pytest.raises(ValueError) as exc_info:
                get_platform_integration_token()

            assert "No platform integration token found" in str(exc_info.value)
            assert "platform_integration_context()" in str(exc_info.value)
