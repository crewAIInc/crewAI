"""OAuth2 device-flow authentication for the CrewAI platform."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypeVar, cast
import webbrowser

import httpx
from pydantic import BaseModel, Field
from rich.console import Console

from crewai_core.auth.utils import validate_jwt_token
from crewai_core.settings import Settings
from crewai_core.token_manager import TokenManager


console = Console()

TOauth2Settings = TypeVar("TOauth2Settings", bound="Oauth2Settings")


class Oauth2Settings(BaseModel):
    """OAuth2 provider configuration."""

    provider: str = Field(
        description="OAuth2 provider used for authentication (e.g., workos, okta, auth0)."
    )
    client_id: str = Field(
        description="OAuth2 client ID issued by the provider, used during authentication requests."
    )
    domain: str = Field(
        description="OAuth2 provider's domain (e.g., your-org.auth0.com) used for issuing tokens."
    )
    audience: str | None = Field(
        description="OAuth2 audience value, typically used to identify the target API or resource.",
        default=None,
    )
    extra: dict[str, Any] = Field(
        description="Extra configuration for the OAuth2 provider.",
        default={},
    )

    @classmethod
    def from_settings(cls: type[TOauth2Settings]) -> TOauth2Settings:
        """Build an ``Oauth2Settings`` instance from the persisted CrewAI settings."""
        settings = Settings()

        return cls(
            provider=settings.oauth2_provider,
            domain=settings.oauth2_domain,
            client_id=settings.oauth2_client_id,
            audience=settings.oauth2_audience,
            extra=settings.oauth2_extra,
        )


if TYPE_CHECKING:
    from crewai_core.auth.providers.base_provider import BaseProvider


class ProviderFactory:
    """Factory for resolving the configured OAuth2 provider."""

    @classmethod
    def from_settings(
        cls: type["ProviderFactory"],  # noqa: UP037
        settings: Oauth2Settings | None = None,
    ) -> "BaseProvider":  # noqa: UP037
        """Create a provider instance from settings, importing the module dynamically."""
        settings = settings or Oauth2Settings.from_settings()

        import importlib

        module = importlib.import_module(
            f"crewai_core.auth.providers.{settings.provider.lower()}"
        )
        provider = getattr(
            module,
            f"{''.join(word.capitalize() for word in settings.provider.split('_'))}Provider",
        )

        return cast("BaseProvider", provider(settings))


class AuthenticationCommand:
    """Drives the OAuth2 device-flow login against the configured provider."""

    def __init__(self) -> None:
        self.token_manager = TokenManager()
        self.oauth2_provider = ProviderFactory.from_settings()

    def login(self) -> None:
        """Sign in to the CrewAI platform via the OAuth2 device flow."""
        console.print("Signing in to CrewAI AMP...\n", style="bold blue")

        device_code_data = self._get_device_code()
        self._display_auth_instructions(device_code_data)

        return self._poll_for_token(device_code_data)

    def _get_device_code(self) -> dict[str, Any]:
        """Request a device code from the provider."""
        device_code_payload = {
            "client_id": self.oauth2_provider.get_client_id(),
            "scope": " ".join(self.oauth2_provider.get_oauth_scopes()),
            "audience": self.oauth2_provider.get_audience(),
        }
        response = httpx.post(
            url=self.oauth2_provider.get_authorize_url(),
            data=device_code_payload,
            timeout=20,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def _display_auth_instructions(self, device_code_data: dict[str, str]) -> None:
        """Print and open the verification URL the user must visit."""
        verification_uri = device_code_data.get(
            "verification_uri_complete", device_code_data.get("verification_uri", "")
        )

        console.print("1. Navigate to: ", verification_uri)
        console.print("2. Enter the following code: ", device_code_data["user_code"])
        webbrowser.open(verification_uri)

    def _poll_for_token(self, device_code_data: dict[str, Any]) -> None:
        """Poll the token endpoint until authentication completes or times out."""
        token_payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code_data["device_code"],
            "client_id": self.oauth2_provider.get_client_id(),
        }

        console.print("\nWaiting for authentication... ", style="bold blue", end="")

        attempts = 0
        while True and attempts < 10:
            response = httpx.post(
                self.oauth2_provider.get_token_url(), data=token_payload, timeout=30
            )
            token_data = response.json()

            if response.status_code == 200:
                self._validate_and_save_token(token_data)

                console.print(
                    "Success!",
                    style="bold green",
                )

                self._post_login()

                console.print("\n[bold green]Welcome to CrewAI AMP![/bold green]\n")
                return

            if token_data["error"] not in ("authorization_pending", "slow_down"):
                raise httpx.HTTPError(
                    token_data.get("error_description") or token_data.get("error")
                )

            time.sleep(device_code_data["interval"])
            attempts += 1

        console.print(
            "Timeout: Failed to get the token. Please try again.", style="bold red"
        )

    def _validate_and_save_token(self, token_data: dict[str, Any]) -> None:
        """Validate the JWT and persist it via the token manager."""
        jwt_token = token_data["access_token"]
        issuer = self.oauth2_provider.get_issuer()
        jwt_token_data = {
            "jwt_token": jwt_token,
            "jwks_url": self.oauth2_provider.get_jwks_url(),
            "issuer": issuer,
            "audience": self.oauth2_provider.get_audience(),
        }

        decoded_token = validate_jwt_token(**jwt_token_data)

        expires_at = decoded_token.get("exp", 0)
        self.token_manager.save_tokens(jwt_token, expires_at)

    def _post_login(self) -> None:
        """Hook called after a successful login. Override to extend behavior."""
