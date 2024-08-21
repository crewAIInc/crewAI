from os import getenv
from typing import Optional

from rich.console import Console

from .api import CrewAPI
from .utils import (
    fetch_and_json_env_file,
    get_auth_token,
    get_git_remote_url,
    get_project_name,
)

console = Console()


class DeployCommand:
    BASE_URL = getenv("BASE_URL", "http://localhost:3000/crewai_plus/api")

    def __init__(self):
        self.project_name = get_project_name()
        self.client = CrewAPI(api_key=get_auth_token())

    def _handle_error(self, json_response: dict) -> None:
        error = json_response.get("error")
        message = json_response.get("message")
        console.print(
            f"Error: {error}",
            style="bold red",
        )
        console.print(
            f"Message: {message}",
            style="bold red",
        )

    def _standard_no_param_error_message(self) -> None:
        console.print(
            "No uuid provided, project pyproject.toml not found or with error.",
            style="bold red",
        )

    def deploy(self, uuid: Optional[str] = None) -> None:
        console.print("Starting deployment...", style="bold blue")
        if uuid:
            response = self.client.deploy_by_uuid(uuid)
        elif self.project_name:
            response = self.client.deploy_by_name(self.project_name)
        else:
            self._standard_no_param_error_message()
            return

        json_response = response.json()
        if response.status_code == 200:
            console.print("Deploying the crew...\n", style="bold blue")

            for key, value in json_response.items():
                console.print(f"{key.title()}: [green]{value}[/green]")

            console.print("\nTo check the status of the deployment, run:")
            console.print("crewai deploy status")
            console.print(" or")
            console.print(f"crewai deploy status --uuid \"{json_response['uuid']}\"")

        else:
            self._handle_error(json_response)

    def create_crew(self) -> None:
        console.print("Creating deployment...", style="bold blue")
        env_vars = fetch_and_json_env_file()
        remote_repo_url = get_git_remote_url()

        input(f"Press Enter to continue with the following Env vars: {env_vars}")
        input(
            f"Press Enter to continue with the following remote repository: {remote_repo_url}\n"
        )
        payload = {
            "deploy": {
                "name": self.project_name,
                "repo_clone_url": remote_repo_url,
                "env": env_vars,
            }
        }

        response = self.client.create_crew(payload)
        if response.status_code == 201:
            json_response = response.json()
            console.print("Deployment created successfully!\n", style="bold green")
            console.print(
                f"Name: {self.project_name} ({json_response['uuid']})",
                style="bold green",
            )
            console.print(f"Status: {json_response['status']}", style="bold green")
            console.print("\nTo (re)deploy the crew, run:")
            console.print("crewai deploy up")
            console.print(" or")
            console.print(f"crewai deploy --uuid {json_response['uuid']}")
        else:
            self._handle_error(response.json())

    def list_crews(self) -> None:
        console.print("Listing all Crews\n", style="bold blue")

        response = self.client.list_crews()
        json_response = response.json()
        if response.status_code == 200:
            for crew_data in json_response:
                console.print(
                    f"- {crew_data['name']} ({crew_data['uuid']}) [blue]{crew_data['status']}[/blue]"
                )
        else:
            console.print(
                "You don't have any crews yet. Let's create one!", style="yellow"
            )
            console.print("  [green]crewai create --name [name][/green]")

    def get_crew_status(self, uuid: Optional[str] = None) -> None:
        console.print("Fetching deployment status...", style="bold blue")
        if uuid:
            response = self.client.status_by_uuid(uuid)
        elif self.project_name:
            response = self.client.status_by_name(self.project_name)
        else:
            self._standard_no_param_error_message()
            return

        json_response = response.json()
        if response.status_code == 200:
            console.print(f"Name:\t {json_response['name']}")
            console.print(f"Status:\t {json_response['status']}")

        else:
            self._handle_error(json_response)

    def get_crew_logs(
        self, uuid: Optional[str], log_type: str = "dExacployment"
    ) -> None:
        console.print(f"Getting {log_type} logs...", style="bold blue")

        if uuid:
            response = self.client.logs_by_uuid(uuid, log_type)
        elif self.project_name:
            response = self.client.logs_by_name(self.project_name, log_type)
        else:
            self._standard_no_param_error_message()
            return

        if response.status_code == 200:
            log_messages = response.json()
            for log_message in log_messages:
                console.print(
                    f"{log_message['timestamp']} - {log_message['level']}: {log_message['message']}"
                )
        else:
            console.print(response.text, style="bold red")

    def remove_crew(self, uuid: Optional[str]) -> None:
        console.print("Removing deployment...", style="bold blue")

        if uuid:
            response = self.client.delete_by_uuid(uuid)
        elif self.project_name:
            response = self.client.delete_by_name(self.project_name)
        else:
            self._standard_no_param_error_message()
            return

        if response.status_code == 204:
            console.print(
                f"Crew '{self.project_name}' removed successfully.", style="green"
            )
        else:
            console.print(
                f"Failed to remove crew '{self.project_name}'", style="bold red"
            )

    def signup(self) -> None:
        console.print("Signing Up", style="bold blue")
        response = self.client.signup()

        # signup_command(response["signup_link"])
        if response.status_code == 200:
            data = response.json()
            console.print(f"Temporary credentials: {data['token']}")
            console.print(
                "We are trying to open the following signup link below.\n"
                "If it doesn't open for you, copy-and-paste into your browser to proceed."
            )
            console.print(f"\n{data['signup_link']}", style="underline blue")
        else:
            console.print(response.text, style="bold red")
