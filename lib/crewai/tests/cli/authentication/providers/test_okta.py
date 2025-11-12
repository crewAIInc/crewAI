import pytest

from crewai.cli.authentication.main import Oauth2Settings
from crewai.cli.authentication.providers.okta import OktaProvider


class TestOktaProvider:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.valid_settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience="test-audience",
        )
        self.provider = OktaProvider(self.valid_settings)

    def test_initialization_with_valid_settings(self):
        provider = OktaProvider(self.valid_settings)
        assert provider.settings == self.valid_settings
        assert provider.settings.provider == "okta"
        assert provider.settings.domain == "test-domain.okta.com"
        assert provider.settings.client_id == "test-client-id"
        assert provider.settings.audience == "test-audience"

    def test_get_authorize_url(self):
        expected_url = "https://test-domain.okta.com/oauth2/default/v1/device/authorize"
        assert self.provider.get_authorize_url() == expected_url

    def test_get_authorize_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="my-company.okta.com",
            client_id="test-client",
            audience="test-audience",
        )
        provider = OktaProvider(settings)
        expected_url = "https://my-company.okta.com/oauth2/default/v1/device/authorize"
        assert provider.get_authorize_url() == expected_url
    
    def test_get_authorize_url_with_custom_authorization_server_name(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": False,
                "authorization_server_name": "my_auth_server_xxxAAA777"
            }
        )
        provider = OktaProvider(settings)
        expected_url = "https://test-domain.okta.com/oauth2/my_auth_server_xxxAAA777/v1/device/authorize"
        assert provider.get_authorize_url() == expected_url

    def test_get_authorize_url_when_using_org_auth_server(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": True,
                "authorization_server_name": None
            }
        )
        provider = OktaProvider(settings)
        expected_url = "https://test-domain.okta.com/oauth2/v1/device/authorize"
        assert provider.get_authorize_url() == expected_url

    def test_get_token_url(self):
        expected_url = "https://test-domain.okta.com/oauth2/default/v1/token"
        assert self.provider.get_token_url() == expected_url

    def test_get_token_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="another-domain.okta.com",
            client_id="test-client",
            audience="test-audience",
        )
        provider = OktaProvider(settings)
        expected_url = "https://another-domain.okta.com/oauth2/default/v1/token"
        assert provider.get_token_url() == expected_url

    def test_get_token_url_with_custom_authorization_server_name(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": False,
                "authorization_server_name": "my_auth_server_xxxAAA777"
            }
        )
        provider = OktaProvider(settings)
        expected_url = "https://test-domain.okta.com/oauth2/my_auth_server_xxxAAA777/v1/token"
        assert provider.get_token_url() == expected_url

    def test_get_token_url_when_using_org_auth_server(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": True,
                "authorization_server_name": None
            }
        )
        provider = OktaProvider(settings)
        expected_url = "https://test-domain.okta.com/oauth2/v1/token"
        assert provider.get_token_url() == expected_url

    def test_get_jwks_url(self):
        expected_url = "https://test-domain.okta.com/oauth2/default/v1/keys"
        assert self.provider.get_jwks_url() == expected_url

    def test_get_jwks_url_with_different_domain(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="dev.okta.com",
            client_id="test-client",
            audience="test-audience",
        )
        provider = OktaProvider(settings)
        expected_url = "https://dev.okta.com/oauth2/default/v1/keys"
        assert provider.get_jwks_url() == expected_url

    def test_get_jwks_url_with_custom_authorization_server_name(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": False,
                "authorization_server_name": "my_auth_server_xxxAAA777"
            }
        )
        provider = OktaProvider(settings)
        expected_url = "https://test-domain.okta.com/oauth2/my_auth_server_xxxAAA777/v1/keys"
        assert provider.get_jwks_url() == expected_url

    def test_get_jwks_url_when_using_org_auth_server(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": True,
                "authorization_server_name": None
            }
        )
        provider = OktaProvider(settings)
        expected_url = "https://test-domain.okta.com/oauth2/v1/keys"
        assert provider.get_jwks_url() == expected_url

    def test_get_issuer(self):
        expected_issuer = "https://test-domain.okta.com/oauth2/default"
        assert self.provider.get_issuer() == expected_issuer

    def test_get_issuer_with_different_domain(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="prod.okta.com",
            client_id="test-client",
            audience="test-audience",
        )
        provider = OktaProvider(settings)
        expected_issuer = "https://prod.okta.com/oauth2/default"
        assert provider.get_issuer() == expected_issuer

    def test_get_issuer_with_custom_authorization_server_name(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": False,
                "authorization_server_name": "my_auth_server_xxxAAA777"
            }
        )
        provider = OktaProvider(settings)
        expected_issuer = "https://test-domain.okta.com/oauth2/my_auth_server_xxxAAA777"
        assert provider.get_issuer() == expected_issuer

    def test_get_issuer_when_using_org_auth_server(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": True,
                "authorization_server_name": None
            }
        )
        provider = OktaProvider(settings)
        expected_issuer = "https://test-domain.okta.com"
        assert provider.get_issuer() == expected_issuer

    def test_get_audience(self):
        assert self.provider.get_audience() == "test-audience"

    def test_get_audience_assertion_error_when_none(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
        )
        provider = OktaProvider(settings)

        with pytest.raises(ValueError, match="Audience is required"):
            provider.get_audience()

    def test_get_client_id(self):
        assert self.provider.get_client_id() == "test-client-id"

    def test_get_required_fields(self):
        assert set(self.provider.get_required_fields()) == set(["authorization_server_name", "using_org_auth_server"]) 
    
    def test_oauth2_base_url(self):
        assert self.provider._oauth2_base_url() == "https://test-domain.okta.com/oauth2/default"
    
    def test_oauth2_base_url_with_custom_authorization_server_name(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": False,
                "authorization_server_name": "my_auth_server_xxxAAA777"
            }
        )

        provider = OktaProvider(settings)
        assert provider._oauth2_base_url() == "https://test-domain.okta.com/oauth2/my_auth_server_xxxAAA777"

    def test_oauth2_base_url_when_using_org_auth_server(self):
        settings = Oauth2Settings(
            provider="okta",
            domain="test-domain.okta.com",
            client_id="test-client-id",
            audience=None,
            extra={
                "using_org_auth_server": True,
                "authorization_server_name": None
            }
        )
        provider = OktaProvider(settings)
        assert provider._oauth2_base_url() == "https://test-domain.okta.com/oauth2"