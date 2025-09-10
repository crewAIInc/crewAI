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
        return self.settings.audience or ""

    def get_client_id(self) -> str:
        assert self.settings.client_id is not None, "Client ID is required"
        return self.settings.client_id

    def _get_domain(self) -> str:
        assert self.settings.domain is not None, "Domain is required"
        return self.settings.domain
