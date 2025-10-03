import pytest
from crewai.cli.authentication.main import Oauth2Settings
from crewai.cli.authentication.providers.auth0 import Auth0Provider



class TestAuth0Provider:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.valid_settings = Oauth2Settings(
            provider="auth0",
            domain="test-domain.auth0.com",
            client_id="test-client-id",
            audience="test-audience"
        )
        self.provider = Auth0Provider(self.valid_settings)

    def test_initialization_with_valid_settings(self):
        provider = Auth0Provider(self.valid_settings)
        assert provider.settings == self.valid_settings
        assert provider.settings.provider == "auth0"
        assert provider.settings.domain == "test-domain.auth0.com"
        assert provider.settings.client_id == "test-client-id"
        assert provider.settings.audience == "test-audience"

    def test_get_authorize_url(self):
        expected_url = "https://test-domain.auth0.com/oauth/device/code"
        assert self.provider.get_authorize_url() == expected_url

    def test_get_authorize_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="auth0",
            domain="my-company.auth0.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = Auth0Provider(settings)
        expected_url = "https://my-company.auth0.com/oauth/device/code"
        assert provider.get_authorize_url() == expected_url

    def test_get_token_url(self):
        expected_url = "https://test-domain.auth0.com/oauth/token"
        assert self.provider.get_token_url() == expected_url

    def test_get_token_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="auth0",
            domain="another-domain.auth0.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = Auth0Provider(settings)
        expected_url = "https://another-domain.auth0.com/oauth/token"
        assert provider.get_token_url() == expected_url

    def test_get_jwks_url(self):
        expected_url = "https://test-domain.auth0.com/.well-known/jwks.json"
        assert self.provider.get_jwks_url() == expected_url

    def test_get_jwks_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="auth0",
            domain="dev.auth0.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = Auth0Provider(settings)
        expected_url = "https://dev.auth0.com/.well-known/jwks.json"
        assert provider.get_jwks_url() == expected_url

    def test_get_issuer(self):
        expected_issuer = "https://test-domain.auth0.com/"
        assert self.provider.get_issuer() == expected_issuer

    def test_get_issuer_with_different_domain(self):
        settings = Oauth2Settings(
            provider="auth0",
            domain="prod.auth0.com",
            client_id="test-client",
            audience="test-audience"
        )
        provider = Auth0Provider(settings)
        expected_issuer = "https://prod.auth0.com/"
        assert provider.get_issuer() == expected_issuer

    def test_get_audience(self):
        assert self.provider.get_audience() == "test-audience"

    def test_get_client_id(self):
        assert self.provider.get_client_id() == "test-client-id"
