from __future__ import annotations

from collections.abc import Callable
import json

from crewai_core.telemetry import Telemetry
import httpx
from rich.console import Console

from crewai_cli.authentication.placeholder import (
    Placeholder,
    PlaceholderSource,
    discard_rejected_placeholder,
    placeholder_client,
    placeholder_for,
)
from crewai_cli.authentication.token import get_auth_token
from crewai_cli.plus_api import PlusAPI


console = Console()


class AuthenticationRequiredError(SystemExit):
    """Raised when a Plus API command needs the user to log in first."""


class PlaceholderRejectedError(AuthenticationRequiredError):
    """AMP answered 401 to the temporary account, which is now discarded."""


def retry_once_on_placeholder_rejection(action: Callable[[], None]) -> None:
    """Run ACTION, and once more when AMP refused the temporary account it used.

    The refused account has expired or was removed and is already discarded,
    so the second run creates its replacement. Only once: a second refusal
    stops the command like any other 401.
    """
    try:
        action()
    except PlaceholderRejectedError:
        action()


class BaseCommand:
    def __init__(self) -> None:
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()


class PlusAPIMixin:
    def __init__(
        self,
        telemetry: Telemetry,
        *,
        base_url: str | None = None,
        placeholder_source: PlaceholderSource | None = None,
    ) -> None:
        """Connect to AMP with the saved login.

        With no login and a PLACEHOLDER_SOURCE, the temporary account for that
        AMP is used instead, created now when a coding agent or a person at a
        terminal runs the command (see `crewai_cli.authentication.placeholder`).
        """
        self.placeholder: Placeholder | None = None
        try:
            telemetry.set_tracer()
            self.plus_api_client = PlusAPI(api_key=get_auth_token(), base_url=base_url)
        except Exception:
            if placeholder_source is not None:
                amp_url = PlusAPI(base_url=base_url).base_url
                self.placeholder = placeholder_for(amp_url, source=placeholder_source)
                if self.placeholder is not None:
                    self.plus_api_client = placeholder_client(self.placeholder, amp_url)
                    return
            telemetry.deploy_signup_error_span()
            console.print(
                "Please sign up/login to CrewAI+ before using the CLI.",
                style="bold red",
            )
            console.print("Run 'crewai login' to sign up/login.", style="bold green")
            raise AuthenticationRequiredError(1) from None

    def _validate_response(self, response: httpx.Response) -> None:
        """Handle and display error messages from API responses.

        Args:
            response: The response from the Plus API.

        Raises:
            PlaceholderRejectedError: AMP answered 401 to the temporary account.
            AuthenticationRequiredError: AMP answered 401 (exit code 1).
            SystemExit: For any other non-2xx response or a body that is not JSON.
        """
        if response.status_code == 401 and self.placeholder is not None:
            discard_rejected_placeholder()
            raise PlaceholderRejectedError(1)
        if response.status_code == 401:
            # AMP also answers 401 for a valid token whose account it does not
            # accept (no AMP user, not activated), which a new login cannot fix:
            # its own message is the only thing that tells the cases apart.
            try:
                body = response.json()
            except ValueError:
                # A proxy or gateway in front of AMP can answer 401 with HTML or nothing.
                body = None
            reason = (
                isinstance(body, dict) and (body.get("message") or body.get("error"))
            ) or response.content.decode("utf-8", errors="replace").strip()
            console.print("CrewAI AMP did not accept your login:", style="bold red")
            if reason:
                console.print(str(reason), markup=False)
            console.print("Run 'crewai login' to sign in again.", style="bold green")
            raise AuthenticationRequiredError(1)

        try:
            json_response = response.json()
        except (json.JSONDecodeError, ValueError):
            console.print(
                "Failed to parse response from Enterprise API failed. Details:",
                style="bold red",
            )
            console.print(f"Status Code: {response.status_code}")
            console.print(
                f"Response:\n{response.content.decode('utf-8', errors='replace')}"
            )
            raise SystemExit(1) from None

        if response.status_code == 422:
            console.print(
                "Failed to complete operation. Please fix the following errors:",
                style="bold red",
            )
            for field, messages in json_response.items():
                for message in messages:
                    console.print(
                        f"* [bold red]{field.capitalize()}[/bold red] {message}"
                    )
            raise SystemExit(1)

        if not response.is_success:
            console.print(
                "Request to Enterprise API failed. Details:", style="bold red"
            )
            details = (
                json_response.get("error")
                or json_response.get("message")
                or response.content.decode("utf-8", errors="replace")
            )
            console.print(f"{details}")
            raise SystemExit(1)
