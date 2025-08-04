from crewai.cli.authentication.constants import (
    WORKOS_DOMAIN,
    WORKOS_CLI_CONNECT_APP_ID,
    WORKOS_ENVIRONMENT_ID,
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
        return self.settings.audience or WORKOS_ENVIRONMENT_ID

    def get_client_id(self) -> str:
        return self.settings.client_id or WORKOS_CLI_CONNECT_APP_ID

    def _get_domain(self) -> str:
        return self.settings.domain or WORKOS_DOMAIN
