from crewai.cli.authentication.providers.base_provider import BaseProvider

class OktaProvider(BaseProvider):
    def get_authorize_url(self) -> str:
        return f"https://{self.settings.domain}/oauth2/default/v1/device/authorize"

    def get_token_url(self) -> str:
        return f"https://{self.settings.domain}/oauth2/default/v1/token"

    def get_jwks_url(self) -> str:
        return f"https://{self.settings.domain}/oauth2/default/v1/keys"

    def get_issuer(self) -> str:
        return f"https://{self.settings.domain}/oauth2/default"

    def get_audience(self) -> str:
        assert self.settings.audience is not None
        return self.settings.audience

    def get_client_id(self) -> str:
        assert self.settings.client_id is not None
        return self.settings.client_id
