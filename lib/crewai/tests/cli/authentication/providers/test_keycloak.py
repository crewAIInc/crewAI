import pytest

from crewai.cli.authentication.main import Oauth2Settings
from crewai.cli.authentication.providers.keycloak import KeycloakProvider


class TestKeycloakProvider:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.valid_settings = Oauth2Settings(
            provider="keycloak",
            domain="keycloak.example.com",
            client_id="test-client-id",
            audience="test-audience",
            extra={
                "realm": "test-realm"
            }
        )
        self.provider = KeycloakProvider(self.valid_settings)

    def test_initialization_with_valid_settings(self):
        provider = KeycloakProvider(self.valid_settings)
        assert provider.settings == self.valid_settings
        assert provider.settings.provider == "keycloak"
        assert provider.settings.domain == "keycloak.example.com"
        assert provider.settings.client_id == "test-client-id"
        assert provider.settings.audience == "test-audience"
        assert provider.settings.extra.get("realm") == "test-realm"

    def test_get_authorize_url(self):
        expected_url = "https://keycloak.example.com/realms/test-realm/protocol/openid-connect/auth/device"
        assert self.provider.get_authorize_url() == expected_url

    def test_get_authorize_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="keycloak",
            domain="auth.company.com",
            client_id="test-client",
            audience="test-audience",
            extra={
                "realm": "my-realm"
            }
        )
        provider = KeycloakProvider(settings)
        expected_url = "https://auth.company.com/realms/my-realm/protocol/openid-connect/auth/device"
        assert provider.get_authorize_url() == expected_url

    def test_get_token_url(self):
        expected_url = "https://keycloak.example.com/realms/test-realm/protocol/openid-connect/token"
        assert self.provider.get_token_url() == expected_url

    def test_get_token_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="keycloak",
            domain="sso.enterprise.com",
            client_id="test-client",
            audience="test-audience",
            extra={
                "realm": "enterprise-realm"
            }
        )
        provider = KeycloakProvider(settings)
        expected_url = "https://sso.enterprise.com/realms/enterprise-realm/protocol/openid-connect/token"
        assert provider.get_token_url() == expected_url

    def test_get_jwks_url(self):
        expected_url = "https://keycloak.example.com/realms/test-realm/protocol/openid-connect/certs"
        assert self.provider.get_jwks_url() == expected_url

    def test_get_jwks_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="keycloak",
            domain="identity.org",
            client_id="test-client",
            audience="test-audience",
            extra={
                "realm": "org-realm"
            }
        )
        provider = KeycloakProvider(settings)
        expected_url = "https://identity.org/realms/org-realm/protocol/openid-connect/certs"
        assert provider.get_jwks_url() == expected_url

    def test_get_issuer(self):
        expected_issuer = "https://keycloak.example.com/realms/test-realm"
        assert self.provider.get_issuer() == expected_issuer

    def test_get_issuer_with_different_domain(self):
        settings = Oauth2Settings(
            provider="keycloak",
            domain="login.myapp.io",
            client_id="test-client",
            audience="test-audience",
            extra={
                "realm": "app-realm"
            }
        )
        provider = KeycloakProvider(settings)
        expected_issuer = "https://login.myapp.io/realms/app-realm"
        assert provider.get_issuer() == expected_issuer

    def test_get_audience(self):
        assert self.provider.get_audience() == "test-audience"

    def test_get_client_id(self):
        assert self.provider.get_client_id() == "test-client-id"

    def test_get_required_fields(self):
        assert self.provider.get_required_fields() == ["realm"]

    def test_oauth2_base_url(self):
        assert self.provider._oauth2_base_url() == "https://keycloak.example.com"

    def test_oauth2_base_url_strips_https_prefix(self):
        settings = Oauth2Settings(
            provider="keycloak",
            domain="https://keycloak.example.com",
            client_id="test-client-id",
            audience="test-audience",
            extra={
                "realm": "test-realm"
            }
        )
        provider = KeycloakProvider(settings)
        assert provider._oauth2_base_url() == "https://keycloak.example.com"

    def test_oauth2_base_url_strips_http_prefix(self):
        settings = Oauth2Settings(
            provider="keycloak",
            domain="http://keycloak.example.com",
            client_id="test-client-id",
            audience="test-audience",
            extra={
                "realm": "test-realm"
            }
        )
        provider = KeycloakProvider(settings)
        assert provider._oauth2_base_url() == "https://keycloak.example.com"
