import time
import webbrowser
from typing import Any, Dict

import requests
from rich.console import Console

from .constants import (
    AUTH0_AUDIENCE,
    AUTH0_CLIENT_ID,
    AUTH0_DOMAIN,
    WORKOS_DOMAIN,
    WORKOS_CLI_CONNECT_APP_ID,
    WORKOS_ENVIRONMENT_ID,
)

from .utils import TokenManager, validate_jwt_token
from urllib.parse import quote
from crewai.cli.plus_api import PlusAPI
from crewai.cli.config import Settings

console = Console()


class AuthenticationCommand:
    AUTH0_DEVICE_CODE_URL = f"https://{AUTH0_DOMAIN}/oauth/device/code"
    AUTH0_TOKEN_URL = f"https://{AUTH0_DOMAIN}/oauth/token"

    WORKOS_DEVICE_CODE_URL = f"https://{WORKOS_DOMAIN}/oauth2/device_authorization"
    WORKOS_TOKEN_URL = f"https://{WORKOS_DOMAIN}/oauth2/token"

    def __init__(self):
        self.token_manager = TokenManager()
        # TODO: WORKOS - This variable is temporary until migration to WorkOS is complete.
        self.user_provider = "workos"

    def login(self) -> None:
        """Sign up to CrewAI+"""

        device_code_url = self.WORKOS_DEVICE_CODE_URL
        token_url = self.WORKOS_TOKEN_URL
        client_id = WORKOS_CLI_CONNECT_APP_ID
        audience = None

        console.print("Signing in to CrewAI Enterprise...\n", style="bold blue")

        # TODO: WORKOS - Next line and conditional are temporary until migration to WorkOS is complete.
        user_provider = self._determine_user_provider()
        if user_provider == "auth0":
            device_code_url = self.AUTH0_DEVICE_CODE_URL
            token_url = self.AUTH0_TOKEN_URL
            client_id = AUTH0_CLIENT_ID
            audience = AUTH0_AUDIENCE
            self.user_provider = "auth0"
        # End of temporary code.

        device_code_data = self._get_device_code(client_id, device_code_url, audience)
        self._display_auth_instructions(device_code_data)

        return self._poll_for_token(device_code_data, client_id, token_url)

    def _get_device_code(
        self, client_id: str, device_code_url: str, audience: str | None = None
    ) -> Dict[str, Any]:
        """Get the device code to authenticate the user."""

        device_code_payload = {
            "client_id": client_id,
            "scope": "openid",
            "audience": audience,
        }
        response = requests.post(
            url=device_code_url, data=device_code_payload, timeout=20
        )
        response.raise_for_status()
        return response.json()

    def _display_auth_instructions(self, device_code_data: Dict[str, str]) -> None:
        """Display the authentication instructions to the user."""
        console.print("1. Navigate to: ", device_code_data["verification_uri_complete"])
        console.print("2. Enter the following code: ", device_code_data["user_code"])
        webbrowser.open(device_code_data["verification_uri_complete"])

    def _poll_for_token(
        self, device_code_data: Dict[str, Any], client_id: str, token_poll_url: str
    ) -> None:
        """Polls the server for the token until it is received, or max attempts are reached."""

        token_payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code_data["device_code"],
            "client_id": client_id,
        }

        console.print("\nWaiting for authentication... ", style="bold blue", end="")

        attempts = 0
        while True and attempts < 10:
            response = requests.post(token_poll_url, data=token_payload, timeout=30)
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
        jwt_token_data = {
            "jwt_token": jwt_token,
            "jwks_url": f"https://{WORKOS_DOMAIN}/oauth2/jwks",
            "issuer": f"https://{WORKOS_DOMAIN}",
            "audience": WORKOS_ENVIRONMENT_ID,
        }

        # TODO: WORKOS - The following conditional is temporary until migration to WorkOS is complete.
        if self.user_provider == "auth0":
            jwt_token_data["jwks_url"] = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
            jwt_token_data["issuer"] = f"https://{AUTH0_DOMAIN}/"
            jwt_token_data["audience"] = AUTH0_AUDIENCE

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
