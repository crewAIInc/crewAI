import time
from typing import TYPE_CHECKING, Any, TypeVar, cast
import webbrowser

from pydantic import BaseModel, Field
import requests
from rich.console import Console

from crewai.cli.authentication.utils import validate_jwt_token
from crewai.cli.config import Settings
from crewai.cli.shared.token_manager import TokenManager


console = Console()

TOauth2Settings = TypeVar("TOauth2Settings", bound="Oauth2Settings")


class Oauth2Settings(BaseModel):
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
        """Create an Oauth2Settings instance from the CLI settings."""

        settings = Settings()

        return cls(
            provider=settings.oauth2_provider,
            domain=settings.oauth2_domain,
            client_id=settings.oauth2_client_id,
            audience=settings.oauth2_audience,
            extra=settings.oauth2_extra,
        )


if TYPE_CHECKING:
    from crewai.cli.authentication.providers.base_provider import BaseProvider


class ProviderFactory:
    @classmethod
    def from_settings(
        cls: type["ProviderFactory"],  # noqa: UP037
        settings: Oauth2Settings | None = None,
    ) -> "BaseProvider":  # noqa: UP037
        settings = settings or Oauth2Settings.from_settings()

        import importlib

        module = importlib.import_module(
            f"crewai.cli.authentication.providers.{settings.provider.lower()}"
        )
        # Converts from snake_case to CamelCase to obtain the provider class name.
        provider = getattr(
            module,
            f"{''.join(word.capitalize() for word in settings.provider.split('_'))}Provider",
        )

        return cast("BaseProvider", provider(settings))


class AuthenticationCommand:
    def __init__(self) -> None:
        self.token_manager = TokenManager()
        self.oauth2_provider = ProviderFactory.from_settings()

    def login(self) -> None:
        """Sign up to CrewAI+"""
        console.print("Signing in to CrewAI AMP...\n", style="bold blue")

        device_code_data = self._get_device_code()
        self._display_auth_instructions(device_code_data)

        return self._poll_for_token(device_code_data)

    def _get_device_code(self) -> dict[str, Any]:
        """Get the device code to authenticate the user."""

        device_code_payload = {
            "client_id": self.oauth2_provider.get_client_id(),
            "scope": " ".join(self.oauth2_provider.get_oauth_scopes()),
            "audience": self.oauth2_provider.get_audience(),
        }
        response = requests.post(
            url=self.oauth2_provider.get_authorize_url(),
            data=device_code_payload,
            timeout=20,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def _display_auth_instructions(self, device_code_data: dict[str, str]) -> None:
        """Display the authentication instructions to the user."""

        verification_uri = device_code_data.get(
            "verification_uri_complete", device_code_data.get("verification_uri", "")
        )

        console.print("1. Navigate to: ", verification_uri)
        console.print("2. Enter the following code: ", device_code_data["user_code"])
        webbrowser.open(verification_uri)

    def _poll_for_token(self, device_code_data: dict[str, Any]) -> None:
        """Polls the server for the token until it is received, or max attempts are reached."""

        token_payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code_data["device_code"],
            "client_id": self.oauth2_provider.get_client_id(),
        }

        console.print("\nWaiting for authentication... ", style="bold blue", end="")

        attempts = 0
        while True and attempts < 10:
            response = requests.post(
                self.oauth2_provider.get_token_url(), data=token_payload, timeout=30
            )
            token_data = response.json()

            if response.status_code == 200:
                self._validate_and_save_token(token_data)

                console.print(
                    "Success!",
                    style="bold green",
                )

                self._login_to_tool_repository()

                console.print("\n[bold green]Welcome to CrewAI AMP![/bold green]\n")
                return

            if token_data["error"] not in ("authorization_pending", "slow_down"):
                raise requests.HTTPError(
                    token_data.get("error_description") or token_data.get("error")
                )

            time.sleep(device_code_data["interval"])
            attempts += 1

        console.print(
            "Timeout: Failed to get the token. Please try again.", style="bold red"
        )

    def _validate_and_save_token(self, token_data: dict[str, Any]) -> None:
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
                f"You are now authenticated to the tool repository for organization [bold cyan]'{settings.org_name if settings.org_name else settings.org_uuid}'[/bold cyan]",
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
