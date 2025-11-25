import pytest

from crewai.cli.authentication.main import Oauth2Settings
from crewai.cli.authentication.providers.entra_id import EntraIdProvider


class TestEntraIdProvider:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.valid_settings = Oauth2Settings(
            provider="entra_id",
            domain="tenant-id-abcdef123456",
            client_id="test-client-id",
            audience="test-audience",
            extra={
                "scope": "openid profile email api://crewai-cli-dev/read"
            }
        )
        self.provider = EntraIdProvider(self.valid_settings)

    def test_initialization_with_valid_settings(self):
        provider = EntraIdProvider(self.valid_settings)
        assert provider.settings == self.valid_settings
        assert provider.settings.provider == "entra_id"
        assert provider.settings.domain == "tenant-id-abcdef123456"
        assert provider.settings.client_id == "test-client-id"
        assert provider.settings.audience == "test-audience"

    def test_get_authorize_url(self):
        expected_url = "https://login.microsoftonline.com/tenant-id-abcdef123456/oauth2/v2.0/devicecode"
        assert self.provider.get_authorize_url() == expected_url

    def test_get_authorize_url_with_different_domain(self):
        # For EntraID, the domain is the tenant ID.
        settings = Oauth2Settings(
            provider="entra_id",
            domain="my-company.entra.id",
            client_id="test-client",
            audience="test-audience",
        )
        provider = EntraIdProvider(settings)
        expected_url = "https://login.microsoftonline.com/my-company.entra.id/oauth2/v2.0/devicecode"
        assert provider.get_authorize_url() == expected_url
    
    def test_get_token_url(self):
        expected_url = "https://login.microsoftonline.com/tenant-id-abcdef123456/oauth2/v2.0/token"
        assert self.provider.get_token_url() == expected_url

    def test_get_token_url_with_different_domain(self):
        # For EntraID, the domain is the tenant ID.
        settings = Oauth2Settings(
            provider="entra_id",
            domain="another-domain.entra.id",
            client_id="test-client",
            audience="test-audience",
        )
        provider = EntraIdProvider(settings)
        expected_url = "https://login.microsoftonline.com/another-domain.entra.id/oauth2/v2.0/token"
        assert provider.get_token_url() == expected_url

    def test_get_jwks_url(self):
        expected_url = "https://login.microsoftonline.com/tenant-id-abcdef123456/discovery/v2.0/keys"
        assert self.provider.get_jwks_url() == expected_url

    def test_get_jwks_url_with_different_domain(self):
        # For EntraID, the domain is the tenant ID.
        settings = Oauth2Settings(
            provider="entra_id",
            domain="dev.entra.id",
            client_id="test-client",
            audience="test-audience",
        )
        provider = EntraIdProvider(settings)
        expected_url = "https://login.microsoftonline.com/dev.entra.id/discovery/v2.0/keys"
        assert provider.get_jwks_url() == expected_url

    def test_get_issuer(self):
        expected_issuer = "https://login.microsoftonline.com/tenant-id-abcdef123456/v2.0"
        assert self.provider.get_issuer() == expected_issuer

    def test_get_issuer_with_different_domain(self):
        # For EntraID, the domain is the tenant ID.
        settings = Oauth2Settings(
            provider="entra_id",
            domain="other-tenant-id-xpto",
            client_id="test-client",
            audience="test-audience",
        )
        provider = EntraIdProvider(settings)
        expected_issuer = "https://login.microsoftonline.com/other-tenant-id-xpto/v2.0"
        assert provider.get_issuer() == expected_issuer

    def test_get_audience(self):
        assert self.provider.get_audience() == "test-audience"

    def test_get_audience_assertion_error_when_none(self):
        settings = Oauth2Settings(
            provider="entra_id",
            domain="test-tenant-id",
            client_id="test-client-id",
            audience=None,
        )
        provider = EntraIdProvider(settings)

        with pytest.raises(ValueError, match="Audience is required"):
            provider.get_audience()

    def test_get_client_id(self):
        assert self.provider.get_client_id() == "test-client-id"

    def test_get_required_fields(self):
        assert set(self.provider.get_required_fields()) == set(["scope"]) 

    def test_get_oauth_scopes(self):
        settings = Oauth2Settings(
            provider="entra_id",
            domain="tenant-id-abcdef123456",
            client_id="test-client-id",
            audience="test-audience",
            extra={
                "scope": "api://crewai-cli-dev/read"
            }
        )
        provider = EntraIdProvider(settings)
        assert provider.get_oauth_scopes() == ["openid", "profile", "email", "api://crewai-cli-dev/read"]
    
    def test_get_oauth_scopes_with_multiple_custom_scopes(self):
        settings = Oauth2Settings(  
            provider="entra_id",
            domain="tenant-id-abcdef123456",
            client_id="test-client-id",
            audience="test-audience",
            extra={
                "scope": "api://crewai-cli-dev/read api://crewai-cli-dev/write custom-scope1 custom-scope2"
            }
        )
        provider = EntraIdProvider(settings)
        assert provider.get_oauth_scopes() == ["openid", "profile", "email", "api://crewai-cli-dev/read", "api://crewai-cli-dev/write", "custom-scope1", "custom-scope2"]

    def test_base_url(self):
        assert self.provider._base_url() == "https://login.microsoftonline.com/tenant-id-abcdef123456"