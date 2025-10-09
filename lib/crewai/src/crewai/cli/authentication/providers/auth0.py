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
        if self.settings.audience is None:
            raise ValueError(
                "Audience is required. Please set it in the configuration."
            )
        return self.settings.audience

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
