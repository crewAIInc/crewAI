# ruff: noqa: S105

import os
from unittest.mock import patch

import pytest
from crewai.context import (
    _platform_integration_token,
    get_platform_integration_token,
    platform_integration_context,
    reset_platform_integration_token,
    set_platform_integration_token,
)


@pytest.fixture
def clean_context():
    """Fixture to ensure clean context state for each test."""
    _platform_integration_token.set(None)
    yield
    _platform_integration_token.set(None)


class TestContextVariableCore:
    """Test core context variable functionality (set/get/reset)."""

    def test_set_and_get_token(self, clean_context):
        """Test basic token setting and retrieval."""
        test_token = "test-token-123"

        assert get_platform_integration_token() is None

        context_token = set_platform_integration_token(test_token)
        assert get_platform_integration_token() == test_token
        assert context_token is not None

    def test_reset_token_restores_previous_state(self, clean_context):
        """Test that reset properly restores previous context state."""
        token1 = "token-1"
        token2 = "token-2"

        context_token1 = set_platform_integration_token(token1)
        assert get_platform_integration_token() == token1

        context_token2 = set_platform_integration_token(token2)
        assert get_platform_integration_token() == token2

        reset_platform_integration_token(context_token2)
        assert get_platform_integration_token() == token1

        reset_platform_integration_token(context_token1)
        assert get_platform_integration_token() is None

    def test_nested_token_management(self, clean_context):
        """Test proper token management with deeply nested contexts."""
        tokens = ["token-1", "token-2", "token-3"]
        context_tokens = []

        for token in tokens:
            context_tokens.append(set_platform_integration_token(token))
            assert get_platform_integration_token() == token

        for i in range(len(tokens) - 1, 0, -1):
            reset_platform_integration_token(context_tokens[i])
            assert get_platform_integration_token() == tokens[i - 1]

        reset_platform_integration_token(context_tokens[0])
        assert get_platform_integration_token() is None

    @patch.dict(os.environ, {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "env-token"})
    def test_context_module_ignores_environment_variables(self, clean_context):
        """Test that context module only returns context values, not env vars."""
        # Context module should not read environment variables
        assert get_platform_integration_token() is None

        # Only context variable should be returned
        set_platform_integration_token("context-token")
        assert get_platform_integration_token() == "context-token"


class TestPlatformIntegrationContext:
    """Test platform integration context manager behavior."""

    def test_basic_context_manager_usage(self, clean_context):
        """Test basic context manager functionality."""
        test_token = "context-token"

        assert get_platform_integration_token() is None

        with platform_integration_context(test_token):
            assert get_platform_integration_token() == test_token

        assert get_platform_integration_token() is None

    @pytest.mark.parametrize("falsy_value", [None, "", False, 0])
    def test_falsy_values_return_nullcontext(self, clean_context, falsy_value):
        """Test that falsy values return nullcontext (no-op)."""
        # Set initial token to verify nullcontext doesn't affect it
        initial_token = "initial-token"
        initial_context_token = set_platform_integration_token(initial_token)

        try:
            with platform_integration_context(falsy_value):
                # Should preserve existing context (nullcontext behavior)
                assert get_platform_integration_token() == initial_token

            # Should still have initial token after nullcontext
            assert get_platform_integration_token() == initial_token
        finally:
            reset_platform_integration_token(initial_context_token)

    @pytest.mark.parametrize("truthy_value", ["token", "123", " ", "0"])
    def test_truthy_values_create_context(self, clean_context, truthy_value):
        """Test that truthy values create proper context."""
        with platform_integration_context(truthy_value):
            assert get_platform_integration_token() == truthy_value

        # Should be cleaned up
        assert get_platform_integration_token() is None

    def test_context_preserves_existing_token(self, clean_context):
        """Test that context manager preserves existing token when exiting."""
        existing_token = "existing-token"
        context_token = "context-token"

        existing_context_token = set_platform_integration_token(existing_token)

        try:
            with platform_integration_context(context_token):
                assert get_platform_integration_token() == context_token

            assert get_platform_integration_token() == existing_token
        finally:
            reset_platform_integration_token(existing_context_token)

    def test_context_manager_return_type(self, clean_context):
        """Test that context manager returns proper types for both cases."""
        # Both should be usable as context managers
        valid_ctx = platform_integration_context("token")
        none_ctx = platform_integration_context(None)

        assert hasattr(valid_ctx, '__enter__')
        assert hasattr(valid_ctx, '__exit__')
        assert hasattr(none_ctx, '__enter__')
        assert hasattr(none_ctx, '__exit__')
