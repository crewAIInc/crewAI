# ruff: noqa: S105

from crewai.context import (
    _platform_integration_token,
    get_platform_integration_token,
)


class TestPlatformIntegrationToken:
    def setup_method(self):
        _platform_integration_token.set(None)

    def teardown_method(self):
        _platform_integration_token.set(None)

    def test_set_and_get(self):
        assert get_platform_integration_token() is None
        _platform_integration_token.set("test-token-123")
        assert get_platform_integration_token() == "test-token-123"

    def test_returns_none_when_not_set(self):
        assert get_platform_integration_token() is None

    def test_overwrite(self):
        _platform_integration_token.set("first")
        _platform_integration_token.set("second")
        assert get_platform_integration_token() == "second"