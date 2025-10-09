# ruff: noqa: S105

import os
from unittest.mock import patch

import pytest
from crewai.context import (
    _platform_integration_token,
    get_platform_integration_token,
    platform_context,
    set_platform_integration_token,
)


class TestPlatformIntegrationToken:
    def setup_method(self):
        _platform_integration_token.set(None)

    def teardown_method(self):
        _platform_integration_token.set(None)

    def test_set_platform_integration_token(self):
        test_token = "test-token-123"

        assert get_platform_integration_token() is None

        set_platform_integration_token(test_token)

        assert get_platform_integration_token() == test_token

    def test_get_platform_integration_token_from_context_var(self):
        test_token = "context-var-token"

        _platform_integration_token.set(test_token)

        assert get_platform_integration_token() == test_token

    @patch.dict(os.environ, {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "env-token-456"})
    def test_get_platform_integration_token_from_env_var(self):
        assert _platform_integration_token.get() is None

        assert get_platform_integration_token() == "env-token-456"

    @patch.dict(os.environ, {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "env-token"})
    def test_context_var_takes_precedence_over_env_var(self):
        context_token = "context-token"

        set_platform_integration_token(context_token)

        assert get_platform_integration_token() == context_token

    @patch.dict(os.environ, {}, clear=True)
    def test_get_platform_integration_token_returns_none_when_not_set(self):
        assert _platform_integration_token.get() is None

        assert get_platform_integration_token() is None

    def test_platform_context_manager_basic_usage(self):
        test_token = "context-manager-token"

        assert get_platform_integration_token() is None

        with platform_context(test_token):
            assert get_platform_integration_token() == test_token

        assert get_platform_integration_token() is None

    def test_platform_context_manager_nested_contexts(self):
        """Test nested platform_context context managers."""
        outer_token = "outer-token"
        inner_token = "inner-token"

        assert get_platform_integration_token() is None

        with platform_context(outer_token):
            assert get_platform_integration_token() == outer_token

            with platform_context(inner_token):
                assert get_platform_integration_token() == inner_token

            assert get_platform_integration_token() == outer_token

        assert get_platform_integration_token() is None

    def test_platform_context_manager_preserves_existing_token(self):
        """Test that platform_context preserves existing token when exiting."""
        initial_token = "initial-token"
        context_token = "context-token"

        set_platform_integration_token(initial_token)
        assert get_platform_integration_token() == initial_token

        with platform_context(context_token):
            assert get_platform_integration_token() == context_token

        assert get_platform_integration_token() == initial_token

    def test_platform_context_manager_exception_handling(self):
        """Test that platform_context properly resets token even when exception occurs."""
        initial_token = "initial-token"
        context_token = "context-token"

        set_platform_integration_token(initial_token)

        with pytest.raises(ValueError):
            with platform_context(context_token):
                assert get_platform_integration_token() == context_token
                raise ValueError("Test exception")

        assert get_platform_integration_token() == initial_token

    def test_platform_context_manager_with_none_initial_state(self):
        """Test platform_context when initial state is None."""
        context_token = "context-token"

        assert get_platform_integration_token() is None

        with pytest.raises(RuntimeError):
            with platform_context(context_token):
                assert get_platform_integration_token() == context_token
                raise RuntimeError("Test exception")

        assert get_platform_integration_token() is None

    @patch.dict(os.environ, {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "env-backup"})
    def test_platform_context_with_env_fallback(self):
        """Test platform_context interaction with environment variable fallback."""
        context_token = "context-token"

        assert get_platform_integration_token() == "env-backup"

        with platform_context(context_token):
            assert get_platform_integration_token() == context_token

        assert get_platform_integration_token() == "env-backup"

    def test_multiple_sequential_context_managers(self):
        """Test multiple sequential uses of platform_context."""
        token1 = "token-1"
        token2 = "token-2"
        token3 = "token-3"

        with platform_context(token1):
            assert get_platform_integration_token() == token1

        assert get_platform_integration_token() is None

        with platform_context(token2):
            assert get_platform_integration_token() == token2

        assert get_platform_integration_token() is None

        with platform_context(token3):
            assert get_platform_integration_token() == token3

        assert get_platform_integration_token() is None

    def test_empty_string_token(self):
        empty_token = ""

        set_platform_integration_token(empty_token)
        assert get_platform_integration_token() == ""

        with platform_context(empty_token):
            assert get_platform_integration_token() == ""

    def test_special_characters_in_token(self):
        special_token = "token-with-!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"

        set_platform_integration_token(special_token)
        assert get_platform_integration_token() == special_token

        with platform_context(special_token):
            assert get_platform_integration_token() == special_token

    def test_very_long_token(self):
        long_token = "a" * 10000

        set_platform_integration_token(long_token)
        assert get_platform_integration_token() == long_token

        with platform_context(long_token):
            assert get_platform_integration_token() == long_token

    @patch.dict(os.environ, {"CREWAI_PLATFORM_INTEGRATION_TOKEN": ""})
    def test_empty_env_var(self):
        assert _platform_integration_token.get() is None
        assert get_platform_integration_token() == ""

    @patch("crewai.context.os.getenv")
    def test_env_var_access_error_handling(self, mock_getenv):
        mock_getenv.side_effect = OSError("Environment access error")

        with pytest.raises(OSError):
            get_platform_integration_token()

    def test_context_var_isolation_between_tests(self):
        """Test that context variable changes don't leak between test methods."""
        test_token = "isolation-test-token"

        assert get_platform_integration_token() is None

        set_platform_integration_token(test_token)
        assert get_platform_integration_token() == test_token

    def test_context_manager_return_value(self):
        """Test that platform_context can be used in with statement with return value."""
        test_token = "return-value-token"

        with platform_context(test_token):
            assert get_platform_integration_token() == test_token

        with platform_context(test_token) as ctx:
            assert ctx is None
            assert get_platform_integration_token() == test_token
