from __future__ import annotations

import json

import httpx
from rich.console import Console

from crewai_cli.authentication.token import get_auth_token
from crewai_cli.plus_api import PlusAPI


console = Console()


class BaseCommand:
    def __init__(self) -> None:
        pass


class PlusAPIMixin:
    def __init__(self) -> None:
        try:
            self.plus_api_client = PlusAPI(api_key=get_auth_token())
        except Exception:
            console.print(
                "Please sign up/login to CrewAI+ before using the CLI.",
                style="bold red",
            )
            console.print("Run 'crewai login' to sign up/login.", style="bold green")
            raise SystemExit from None

    def _validate_response(self, response: httpx.Response) -> None:
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
            raise SystemExit from None

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
            raise SystemExit

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
            raise SystemExit
