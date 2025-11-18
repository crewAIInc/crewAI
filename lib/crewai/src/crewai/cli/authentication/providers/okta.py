from crewai.cli.authentication.providers.base_provider import BaseProvider


class OktaProvider(BaseProvider):
    def get_authorize_url(self) -> str:
        return f"{self._oauth2_base_url()}/v1/device/authorize"

    def get_token_url(self) -> str:
        return f"{self._oauth2_base_url()}/v1/token"

    def get_jwks_url(self) -> str:
        return f"{self._oauth2_base_url()}/v1/keys"

    def get_issuer(self) -> str:
        return self._oauth2_base_url().removesuffix("/oauth2")

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

    def get_required_fields(self) -> list[str]:
        return ["authorization_server_name", "using_org_auth_server"]

    def _oauth2_base_url(self) -> str:
        using_org_auth_server = self.settings.extra.get("using_org_auth_server", False)

        if using_org_auth_server:
            base_url = f"https://{self.settings.domain}/oauth2"
        else:
            base_url = f"https://{self.settings.domain}/oauth2/{self.settings.extra.get('authorization_server_name', 'default')}"

        return f"{base_url}"
