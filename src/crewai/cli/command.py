from typing import Dict, Any
from rich.console import Console
from crewai.cli.plus_api import PlusAPI
from crewai.cli.utils import get_auth_token
from crewai.telemetry.telemetry import Telemetry

console = Console()


class BaseCommand:
    def __init__(self):
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()


class PlusAPIMixin:
    def __init__(self, telemetry):
        try:
            telemetry.set_tracer()
            self.plus_api_client = PlusAPI(api_key=get_auth_token())
        except Exception:
            self._deploy_signup_error_span = telemetry.deploy_signup_error_span()
            console.print(
                "Please sign up/login to CrewAI+ before using the CLI.",
                style="bold red",
            )
            console.print("Run 'crewai signup' to sign up/login.", style="bold green")
            raise SystemExit

    def _handle_plus_api_error(self, json_response: Dict[str, Any]) -> None:
        """
        Handle and display error messages from API responses.

        Args:
            json_response (Dict[str, Any]): The JSON response containing error information.
        """
        error = json_response.get("error", "Unknown error")
        message = json_response.get("message", "No message provided")
        console.print(f"Error: {error}", style="bold red")
        console.print(f"Message: {message}", style="bold red")
