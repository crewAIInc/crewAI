import base64
import hashlib
import secrets
import textwrap
import webbrowser
import socket
from urllib.parse import parse_qs, urlparse
import requests
from rich.console import Console
from crewai.cli.plus_api import PlusAPI

from crewai.cli.tools.main import ToolCommand

from .constants import (
    WORKOS_AUTHORIZE_URL,
    WORKOS_CLIENT_ID,
    WORKOS_TOKEN_URL,
    AUTH0_AUDIENCE,  # Legacy Import for old auth
    AUTH0_CLIENT_ID,  # Legacy Import for old auth
    AUTH0_DOMAIN,  # Legacy Import for old auth
)

from typing import Any, Dict  # Legacy Import for old auth
import time  # Legacy Import for old auth


from .utils import (
    TokenManager,
    validate_token,
    old_validate_token,  # Legacy Import for old auth
)

console = Console()


class AuthenticationCommand:
    CODE_VERIFIER = secrets.token_urlsafe(64)
    CODE_CHALLENGE = (
        base64.urlsafe_b64encode(hashlib.sha256(CODE_VERIFIER.encode()).digest())
        .rstrip(b"=")
        .decode("utf-8")
    )
    NONCE = secrets.token_hex(6)
    STATE = secrets.token_hex(9)
    SOCKET_HOST = "0.0.0.0"  # nosec: B104
    SOCKET_PORT = 49152

    # Legacy authentication cosntants
    DEVICE_CODE_URL = f"https://{AUTH0_DOMAIN}/oauth/device/code"
    TOKEN_URL = f"https://{AUTH0_DOMAIN}/oauth/token"

    def __init__(self):
        self.token_manager = TokenManager()
        self.auth_url = self._get_auth_url()

    # TODO: While we migrate from auth providers, we should use this method to determine which auth provider to use.
    # Afterwards, we can remove this method and the old_login method and change new_login to login.
    def login(self) -> None:
        """Login or Sign Up to CrewAI Enterprise"""

        email = input("Enter your email: ")
        response = PlusAPI("").get_provider(email)

        if response.status_code == 200:
            if response.json()["provider"] == "auth0":
                self.old_login()
            else:
                self.new_login()
        else:
            console.print(
                "Error: Failed to authenticate with crewai enterprise. Ensure that you are using the latest crewai version and please try again. If the problem persists, contact support@crewai.com.",
                style="red",
            )
            raise SystemExit

    def new_login(self) -> None:
        """Login or Sign Up to CrewAI Enterprise"""

        console.print("Signing in to CrewAI enterprise... \n", style="bold blue")

        # 1. Get the auth URL. Upon successful authentication, browser will redirect back to CLI with the 'code' parameter.
        console.print(
            f"1. Navigate to [bold blue][link={self.auth_url}]this link.[/link][/bold blue] (it should open automatically in a few seconds...)",
            style="bold",
        )
        webbrowser.open(self.auth_url)

        # 2. Listen for the auth response from the browser, and upon receiving the 'code' parameter, authenticate the user.
        redirect_url_params = self._listen_for_auth_response()
        console.print(
            "2. Login successful. Retrieving your [bold blue]access tokens[/bold blue]...",
            style="bold",
        )
        auth_response = self._authenticate(redirect_url_params)

        # 3. Validate the JWT token signature, extract the access and refresh tokens and save them to the token manager.
        access_token, refresh_token, user_info = self._validate_and_extract_tokens(
            auth_response
        )
        self.token_manager.save_access_token(access_token, auth_response["expires_in"])
        self.token_manager.save_refresh_token(refresh_token)

        # 4. Sign in to the tool repository.
        console.print(
            "3. All good. Now signing you in to [bold blue]tool repository[/bold blue]...",
            style="bold",
        )
        self._sign_in_to_tool_repository()

        # 5. Wrap up.
        console.print(
            f"4. Done! You are now logged in.\n\n Welcome to CrewAI enterprise, [bold cyan]{user_info.get('name')}[/bold cyan].",
            style="bold green",
        )
        return None

    def _get_auth_url(self) -> str:
        return (
            f"{WORKOS_AUTHORIZE_URL}?"
            f"response_type=code&"
            f"client_id={WORKOS_CLIENT_ID}&"
            f"redirect_uri=http://localhost:{self.SOCKET_PORT}&"
            f"scope=openid+profile+email+offline_access&"
            f"code_challenge={self.CODE_CHALLENGE}&"
            f"code_challenge_method=S256&"
            f"nonce={self.NONCE}&"
            f"state={self.STATE}"
        )

    def _listen_for_auth_response(self) -> dict[str, str | list[str]]:
        """
        Listen for the authentication response from the browser.

        Returns:
            dict[str, str]: The URL parameters passed in the querystring of the redirect URL.
        """

        redirect_url_params: dict[str, str | list[str]] = {}

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.SOCKET_HOST, self.SOCKET_PORT))
            server_socket.listen(1)
            console.print("> Waiting for browser login and approval...", style="yellow")

            conn, addr = server_socket.accept()
            with conn:
                request = conn.recv(1024).decode("utf-8")

                # Extract the request line (first line of the HTTP request)
                request_line = request.splitlines()[0]
                method, path, _ = request_line.split()

                # Parse the URL path to get query string parameters
                parsed_url = urlparse(path)

                # Convert values from lists to single values if appropriate
                redirect_url_params = {
                    k: v[0] if len(v) == 1 else v
                    for k, v in parse_qs(parsed_url.query).items()
                }

                # Prepare the HTTP response with success message and JS that attempts to close the tab.
                html_body = self._html_response_body()
                http_response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(html_body.encode('utf-8'))}\r\nConnection: close\r\n\r\n{html_body}"
                conn.sendall(http_response.encode("utf-8"))

            server_socket.close()
            console.print("> Response received. Proceeding to login...", style="green")

        return redirect_url_params

    def _authenticate(self, params) -> dict[str, str]:
        response = requests.post(
            WORKOS_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": WORKOS_CLIENT_ID,
                "code": params["code"],
                "redirect_uri": f"http://localhost:{self.SOCKET_PORT}",
                "code_verifier": self.CODE_VERIFIER,
            },
            timeout=10,
        )

        if response.status_code != 200:
            console.print(
                "❌ Failed to sign in to CrewAI enterprise. \nRun [bold]crewai login[/bold] and try logging in again.\n",
                style="red",
            )
            raise SystemExit

        return response.json()

    def _validate_and_extract_tokens(
        self, response_dict: dict[str, str]
    ) -> tuple[str, str, dict[str, str]]:
        user_info = {}
        try:
            validate_token(response_dict["access_token"])
            user_info = validate_token(response_dict["id_token"], "id_token")
        except Exception as e:
            console.print(
                f"❌ Failure validating JWT token signature, login failed. \nRun [bold]crewai login[/bold] to try logging in again.\n\n Error: {e}",
                style="red",
            )
            raise SystemExit

        return response_dict["access_token"], response_dict["refresh_token"], user_info

    def _sign_in_to_tool_repository(self) -> None:
        try:
            ToolCommand().login()
        except Exception as e:
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
            console.print(f"Error: {e}", style="red")

    def _html_response_body(self) -> str:
        html_body = textwrap.dedent("""\
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Authentication Successful</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background-color: #f0f2f5; text-align: center; }
                    .container { background-color: #ffffff; padding: 30px 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }
                    .logo { max-height: 4rem; }
                    h1 { color: #FF5B50; font-size: 24px; margin-bottom: 25px; }
                    p { color: #333; font-size: 16px; line-height: 1.5; }
                </style>
                <script type="text/javascript">
                    window.onload = function() {
                        // Attempt to close the window/tab.
                        // Browsers may block this if the window was not opened by a script from this origin's perspective.
                        window.close();
                    };
                </script>
            </head>
            <body>
                <div class="container">
                    <img class="logo" src="data:image/png;base64,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">
                    <h1>Authentication Successful!</h1>
                    <p>Your authentication through CLI was successful. This browser tab should close automatically.</p>
                    <p>If it doesn't close in a few seconds, you may close it manually.</p>
                </div>
            </body>
            </html>""")
        return html_body

    # Legacy auth methods below
    def old_login(self) -> None:
        """Sign up to CrewAI+"""
        console.print("Signing Up to CrewAI+ \n", style="bold blue")
        device_code_data = self._old_get_device_code()
        self._old_display_auth_instructions(device_code_data)

        return self._old_poll_for_token(device_code_data)

    def _old_get_device_code(self) -> Dict[str, Any]:
        """Get the device code to authenticate the user."""

        device_code_payload = {
            "client_id": AUTH0_CLIENT_ID,
            "scope": "openid",
            "audience": AUTH0_AUDIENCE,
        }
        response = requests.post(
            url=self.DEVICE_CODE_URL, data=device_code_payload, timeout=20
        )
        response.raise_for_status()
        return response.json()

    def _old_display_auth_instructions(self, device_code_data: Dict[str, str]) -> None:
        """Display the authentication instructions to the user."""
        console.print("1. Navigate to: ", device_code_data["verification_uri_complete"])
        console.print("2. Enter the following code: ", device_code_data["user_code"])
        webbrowser.open(device_code_data["verification_uri_complete"])

    def _old_poll_for_token(self, device_code_data: Dict[str, Any]) -> None:
        """Poll the server for the token."""
        token_payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code_data["device_code"],
            "client_id": AUTH0_CLIENT_ID,
        }

        attempts = 0
        while True and attempts < 5:
            response = requests.post(self.TOKEN_URL, data=token_payload, timeout=30)
            token_data = response.json()

            if response.status_code == 200:
                old_validate_token(token_data["id_token"])
                expires_in = 360000  # Token expiration time in seconds
                self.token_manager.save_access_token(
                    token_data["access_token"], expires_in
                )

                try:
                    ToolCommand().login()
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
