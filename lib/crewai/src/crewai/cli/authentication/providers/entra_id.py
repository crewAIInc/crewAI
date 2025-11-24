from typing import cast

from crewai.cli.authentication.providers.base_provider import BaseProvider


class EntraIdProvider(BaseProvider):
    def get_authorize_url(self) -> str:
        return f"{self._base_url()}/oauth2/v2.0/devicecode"

    def get_token_url(self) -> str:
        return f"{self._base_url()}/oauth2/v2.0/token"

    def get_jwks_url(self) -> str:
        return f"{self._base_url()}/discovery/v2.0/keys"

    def get_issuer(self) -> str:
        return f"{self._base_url()}/v2.0"

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

    def get_oauth_scopes(self) -> list[str]:
        return [
            *super().get_oauth_scopes(),
            *cast(str, self.settings.extra.get("scope", "")).split(),
        ]

    def get_required_fields(self) -> list[str]:
        return ["scope"]

    def _base_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.settings.domain}"
