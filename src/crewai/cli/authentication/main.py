import time
import webbrowser
from typing import Any, Dict

import requests
from rich.console import Console

from .constants import AUTH0_AUDIENCE, AUTH0_CLIENT_ID, AUTH0_DOMAIN
from .utils import TokenManager, validate_token

console = Console()


class AuthenticationCommand:
    DEVICE_CODE_URL = f"https://{AUTH0_DOMAIN}/oauth/device/code"
    TOKEN_URL = f"https://{AUTH0_DOMAIN}/oauth/token"

    def __init__(self):
        self.token_manager = TokenManager()

    def signup(self) -> None:
        """Sign up to CrewAI+"""
        console.print("Signing Up to CrewAI+ \n", style="bold blue")
        device_code_data = self._get_device_code()
        self._display_auth_instructions(device_code_data)

        return self._poll_for_token(device_code_data)

    def _get_device_code(self) -> Dict[str, Any]:
        """Get the device code to authenticate the user."""

        device_code_payload = {
            "client_id": AUTH0_CLIENT_ID,
            "scope": "openid",
            "audience": AUTH0_AUDIENCE,
        }
        response = requests.post(url=self.DEVICE_CODE_URL, data=device_code_payload)
        response.raise_for_status()
        return response.json()

    def _display_auth_instructions(self, device_code_data: Dict[str, str]) -> None:
        """Display the authentication instructions to the user."""
        console.print("1. Navigate to: ", device_code_data["verification_uri_complete"])
        console.print("2. Enter the following code: ", device_code_data["user_code"])
        webbrowser.open(device_code_data["verification_uri_complete"])

    def _poll_for_token(self, device_code_data: Dict[str, Any]) -> None:
        """Poll the server for the token."""
        token_payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code_data["device_code"],
            "client_id": AUTH0_CLIENT_ID,
        }

        attempts = 0
        while True and attempts < 5:
            response = requests.post(self.TOKEN_URL, data=token_payload)
            token_data = response.json()

            if response.status_code == 200:
                validate_token(token_data["id_token"])
                expires_in = 360000  # Token expiration time in seconds
                self.token_manager.save_tokens(token_data["access_token"], expires_in)
                console.print("\nWelcome to CrewAI+ !!", style="green")
                return

            if token_data["error"] not in ("authorization_pending", "slow_down"):
                raise requests.HTTPError(token_data["error_description"])

            time.sleep(device_code_data["interval"])
            attempts += 1

        console.print(
            "Timeout: Failed to get the token. Please try again.", style="bold red"
        )
