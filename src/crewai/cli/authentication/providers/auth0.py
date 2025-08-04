from crewai.cli.authentication.constants import (
    AUTH0_AUDIENCE,
    AUTH0_CLIENT_ID,
    AUTH0_DOMAIN,
)
from crewai.cli.authentication.providers.base_provider import BaseProvider

class Auth0Provider(BaseProvider):
    def get_authorize_url(self) -> str:
        return f"https://{self._get_domain()}/oauth/device/code"

    def get_token_url(self) -> str:
        return f"https://{self._get_domain()}/oauth/token"

    def get_jwks_url(self) -> str:
        return f"https://{self._get_domain()}/.well-known/jwks.json"

    def get_issuer(self) -> str:
        return f"https://{self._get_domain()}/"

    def get_audience(self) -> str:
        return self.settings.audience or AUTH0_AUDIENCE

    def get_client_id(self) -> str:
        return self.settings.client_id or AUTH0_CLIENT_ID

    def _get_domain(self) -> str:
        return self.settings.domain or AUTH0_DOMAIN
