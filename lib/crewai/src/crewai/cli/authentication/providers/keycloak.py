from crewai.cli.authentication.providers.base_provider import BaseProvider


class KeycloakProvider(BaseProvider):
    def get_authorize_url(self) -> str:
        return f"{self._oauth2_base_url()}/realms/{self.settings.extra.get('realm')}/protocol/openid-connect/auth/device"

    def get_token_url(self) -> str:
        return f"{self._oauth2_base_url()}/realms/{self.settings.extra.get('realm')}/protocol/openid-connect/token"

    def get_jwks_url(self) -> str:
        return f"{self._oauth2_base_url()}/realms/{self.settings.extra.get('realm')}/protocol/openid-connect/certs"

    def get_issuer(self) -> str:
        return f"{self._oauth2_base_url()}/realms/{self.settings.extra.get('realm')}"

    def get_audience(self) -> str:
        return self.settings.audience or "no-audience-provided"

    def get_client_id(self) -> str:
        if self.settings.client_id is None:
            raise ValueError(
                "Client ID is required. Please set it in the configuration."
            )
        return self.settings.client_id

    def get_required_fields(self) -> list[str]:
        return ["realm"]

    def _oauth2_base_url(self) -> str:
        domain = self.settings.domain.removeprefix("https://").removeprefix("http://")
        return f"https://{domain}"
