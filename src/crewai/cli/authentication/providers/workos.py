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
        if self.settings.client_id is None:
            raise ValueError(
                "Client ID is required. Please set it in the configuration."
            )
        return self.settings.client_id

    def _get_domain(self) -> str:
        if self.settings.domain is None:
            raise ValueError("Domain is required. Please set it in the configuration.")
        return self.settings.domain
