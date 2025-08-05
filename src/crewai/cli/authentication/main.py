import time
import webbrowser
from typing import Any, Dict, Optional

import requests
from rich.console import Console
from pydantic import BaseModel, Field


from .utils import TokenManager, validate_jwt_token
from urllib.parse import quote
from crewai.cli.plus_api import PlusAPI
from crewai.cli.config import Settings
from crewai.cli.authentication.constants import (
    AUTH0_AUDIENCE,
    AUTH0_CLIENT_ID,
    AUTH0_DOMAIN,
)

console = Console()


class Oauth2Settings(BaseModel):
    provider: str = Field(description="OAuth2 provider used for authentication (e.g., workos, okta, auth0).")
    client_id: str = Field(description="OAuth2 client ID issued by the provider, used during authentication requests.")
    domain: str = Field(description="OAuth2 provider's domain (e.g., your-org.auth0.com) used for issuing tokens.")
    audience: Optional[str] = Field(description="OAuth2 audience value, typically used to identify the target API or resource.", default=None)

    @classmethod
    def from_settings(cls):
        settings = Settings()

        return cls(
            provider=settings.oauth2_provider,
            domain=settings.oauth2_domain,
            client_id=settings.oauth2_client_id,
            audience=settings.oauth2_audience,
        )


class ProviderFactory:
    @classmethod
    def from_settings(cls, settings: Optional[Oauth2Settings] = None):
        settings = settings or Oauth2Settings.from_settings()

        import importlib
        module = importlib.import_module(f"crewai.cli.authentication.providers.{settings.provider.lower()}")
        provider = getattr(module, f"{settings.provider.capitalize()}Provider")

        return provider(settings)

class AuthenticationCommand:
    def __init__(self):
        self.token_manager = TokenManager()
        self.oauth2_provider = ProviderFactory.from_settings()

    def login(self) -> None:
        """Sign up to CrewAI+"""
        console.print("Signing in to CrewAI Enterprise...\n", style="bold blue")

        # TODO: WORKOS - Next line and conditional are temporary until migration to WorkOS is complete.
        user_provider = self._determine_user_provider()
        if user_provider == "auth0":
            settings = Oauth2Settings(
                provider="auth0",
                client_id=AUTH0_CLIENT_ID,
                domain=AUTH0_DOMAIN,
                audience=AUTH0_AUDIENCE
            )
            self.oauth2_provider = ProviderFactory.from_settings(settings)
        # End of temporary code.

        device_code_data = self._get_device_code()
        self._display_auth_instructions(device_code_data)

        return self._poll_for_token(device_code_data)

    def _get_device_code(
        self
    ) -> Dict[str, Any]:
        """Get the device code to authenticate the user."""

        device_code_payload = {
            "client_id": self.oauth2_provider.get_client_id(),
            "scope": "openid",
            "audience": self.oauth2_provider.get_audience(),
        }
        response = requests.post(
            url=self.oauth2_provider.get_authorize_url(), data=device_code_payload, timeout=20
        )
        response.raise_for_status()
        return response.json()

    def _display_auth_instructions(self, device_code_data: Dict[str, str]) -> None:
        """Display the authentication instructions to the user."""
        console.print("1. Navigate to: ", device_code_data["verification_uri_complete"])
        console.print("2. Enter the following code: ", device_code_data["user_code"])
        webbrowser.open(device_code_data["verification_uri_complete"])

    def _poll_for_token(
        self, device_code_data: Dict[str, Any]
    ) -> None:
        """Polls the server for the token until it is received, or max attempts are reached."""

        token_payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code_data["device_code"],
            "client_id": self.oauth2_provider.get_client_id(),
        }

        console.print("\nWaiting for authentication... ", style="bold blue", end="")

        attempts = 0
        while True and attempts < 10:
            response = requests.post(self.oauth2_provider.get_token_url(), data=token_payload, timeout=30)
            token_data = response.json()

            if response.status_code == 200:
                self._validate_and_save_token(token_data)

                console.print(
                    "Success!",
                    style="bold green",
                )

                self._login_to_tool_repository()

                console.print(
                    "\n[bold green]Welcome to CrewAI Enterprise![/bold green]\n"
                )
                return

            if token_data["error"] not in ("authorization_pending", "slow_down"):
                raise requests.HTTPError(token_data["error_description"])

            time.sleep(device_code_data["interval"])
            attempts += 1

        console.print(
            "Timeout: Failed to get the token. Please try again.", style="bold red"
        )

    def _validate_and_save_token(self, token_data: Dict[str, Any]) -> None:
        """Validates the JWT token and saves the token to the token manager."""

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

    def _login_to_tool_repository(self) -> None:
        """Login to the tool repository."""

        from crewai.cli.tools.main import ToolCommand

        try:
            console.print(
                "Now logging you in to the Tool Repository... ",
                style="bold blue",
                end="",
            )

            ToolCommand().login()

            console.print(
                "Success!\n",
                style="bold green",
            )

            settings = Settings()
            console.print(
                f"You are authenticated to the tool repository as [bold cyan]'{settings.org_name}'[/bold cyan] ({settings.org_uuid})",
                style="green",
            )
        except Exception:
            console.print(
                "\n[bold yellow]Warning:[/bold yellow] Authentication with the Tool Repository failed.",
                style="yellow",
            )
            console.print(
                "Other features will work normally, but you may experience limitations "
                "with downloading and publishing tools."
                "\nRun [bold]crewai login[/bold] to try logging in again.\n",
                style="yellow",
            )

    # TODO: WORKOS - This method is temporary until migration to WorkOS is complete.
    def _determine_user_provider(self) -> str:
        """Determine which provider to use for authentication."""

        console.print(
            "Enter your CrewAI Enterprise account email: ", style="bold blue", end=""
        )
        email = input()
        email_encoded = quote(email)

        # It's not correct to call this method directly, but it's temporary until migration is complete.
        response = PlusAPI("")._make_request(
            "GET", f"/crewai_plus/api/v1/me/provider?email={email_encoded}"
        )

        if response.status_code == 200:
            if response.json().get("provider") == "auth0":
                return "auth0"
            else:
                return "workos"
        else:
            console.print(
                "Error: Failed to authenticate with crewai enterprise. Ensure that you are using the latest crewai version and please try again. If the problem persists, contact support@crewai.com.",
                style="red",
            )
            raise SystemExit
