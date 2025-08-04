from crewai.cli.constants import (
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
)
from crewai.cli.authentication.providers.base_provider import BaseProvider

class WorkosProvider(BaseProvider):
    def get_authorize_url(self) -> str:
        return f"https://{self._get_domain()}/oauth2/device_authorization"

    def get_token_url(self) -> str:
        return f"https://{self._get_domain()}/oauth2/token"

    def get_jwks_url(self) -> str:
        return f"https://{self._get_domain()}/oauth2/jwks"

    def get_issuer(self) -> str:
        return f"https://{self._get_domain()}"

    def get_audience(self) -> str:
        return self.settings.audience or CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE

    def get_client_id(self) -> str:
        return self.settings.client_id or CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID

    def _get_domain(self) -> str:
        return self.settings.domain or CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN
