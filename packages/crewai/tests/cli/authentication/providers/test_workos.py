import pytest
from crewai.cli.authentication.main import Oauth2Settings
from crewai.cli.authentication.providers.workos import WorkosProvider


class TestWorkosProvider:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.valid_settings = Oauth2Settings(
            provider="workos",
            domain="login.company.com",
            client_id="test-client-id",
            audience="test-audience"
        )
        self.provider = WorkosProvider(self.valid_settings)

    def test_initialization_with_valid_settings(self):
        provider = WorkosProvider(self.valid_settings)
        assert provider.settings == self.valid_settings
        assert provider.settings.provider == "workos"
        assert provider.settings.domain == "login.company.com"
        assert provider.settings.client_id == "test-client-id"
        assert provider.settings.audience == "test-audience"

    def test_get_authorize_url(self):
        expected_url = "https://login.company.com/oauth2/device_authorization"
        assert self.provider.get_authorize_url() == expected_url

    def test_get_authorize_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="workos",
            domain="login.example.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = WorkosProvider(settings)
        expected_url = "https://login.example.com/oauth2/device_authorization"
        assert provider.get_authorize_url() == expected_url

    def test_get_token_url(self):
        expected_url = "https://login.company.com/oauth2/token"
        assert self.provider.get_token_url() == expected_url

    def test_get_token_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="workos",
            domain="api.workos.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = WorkosProvider(settings)
        expected_url = "https://api.workos.com/oauth2/token"
        assert provider.get_token_url() == expected_url

    def test_get_jwks_url(self):
        expected_url = "https://login.company.com/oauth2/jwks"
        assert self.provider.get_jwks_url() == expected_url

    def test_get_jwks_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="workos",
            domain="auth.enterprise.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = WorkosProvider(settings)
        expected_url = "https://auth.enterprise.com/oauth2/jwks"
        assert provider.get_jwks_url() == expected_url

    def test_get_issuer(self):
        expected_issuer = "https://login.company.com"
        assert self.provider.get_issuer() == expected_issuer

    def test_get_issuer_with_different_domain(self):
        settings = Oauth2Settings(
            provider="workos",
            domain="sso.company.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = WorkosProvider(settings)
        expected_issuer = "https://sso.company.com"
        assert provider.get_issuer() == expected_issuer

    def test_get_audience(self):
        assert self.provider.get_audience() == "test-audience"

    def test_get_audience_fallback_to_default(self):
        settings = Oauth2Settings(
            provider="workos",
            domain="login.company.com",
            client_id="test-client-id",
            audience=None
        )
        provider = WorkosProvider(settings)
        assert provider.get_audience() == ""

    def test_get_client_id(self):
        assert self.provider.get_client_id() == "test-client-id"
