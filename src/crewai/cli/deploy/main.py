from os import getenv

import requests
from rich.console import Console

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
        self.remote_repo_url = get_git_remote_url()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {get_auth_token()}",
            "Content-Type": "application/json",
        }
        return requests.request(method, url, headers=headers, **kwargs)

    def deploy(self) -> None:
        console.print("Deploying the crew...", style="bold blue")
        response = self._make_request(
            "POST", f"crews/by-name/{self.project_name}/deploy"
        )
        console.print(response.json())

    def create_crew(self) -> None:
        console.print("Creating deployment...", style="bold blue")
        env_vars = fetch_and_json_env_file()
        payload = {
            "deploy": {
                "name": self.project_name,
                "repo_clone_url": self.remote_repo_url,
                "env": env_vars,
            }
        }
        response = self._make_request("POST", "crews", json=payload)
        console.print(response.json())

    def list_crews(self) -> None:
        console.print("Listing all Crews", style="bold blue")
        response = self._make_request("GET", "crews")
        crews_data = response.json()

        if response.status_code == 200:
            if crews_data:
                for crew_data in crews_data:
                    console.print(
                        f"- {crew_data['name']} ({crew_data['uuid']}) [blue]{crew_data['status']}[/blue]"
                    )
            else:
                console.print(
                    "You don't have any crews yet. Let's create one!", style="yellow"
                )
                console.print("\t[green]crewai create --name [name][/green]")

    def get_crew_status(self) -> None:
        console.print("Getting deployment status...", style="bold blue")
        response = self._make_request(
            "GET", f"crews/by-name/{self.project_name}/status"
        )

        if response.status_code == 200:
            status_data = response.json()
            console.print(f"Name:\t {status_data['name']}")
            console.print(f"Status:\t {status_data['status']}")
            console.print("\nUsage:")
            console.print(f"\tcrewai inputs --name \"{status_data['name']}\"")
            console.print(
                f"\tcrewai kickoff --name \"{status_data['name']}\" --inputs [INPUTS]"
            )
        else:
            console.print(response.json(), style="bold red")

    def get_crew_logs(self, log_type: str = "deployment") -> None:
        console.print("Getting deployment logs...", style="bold blue")
        response = self._make_request(
            "GET", f"crews/by-name/{self.project_name}/logs/{log_type}"
        )

        if response.status_code == 200:
            log_messages = response.json()
            for log_message in log_messages:
                console.print(
                    f"{log_message['timestamp']} - {log_message['level']}: {log_message['message']}"
                )
        else:
            console.print(response.text, style="bold red")

    def remove_crew(self) -> None:
        console.print("Removing deployment...", style="bold blue")
        response = self._make_request("DELETE", f"crews/by-name/{self.project_name}")

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
        response = self._make_request("GET", "signup_link")

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
