from typing import Any, Dict, List, Optional

from rich.console import Console

from crewai.cli import git
from crewai.cli.command import BaseCommand, PlusAPIMixin
from crewai.cli.utils import fetch_and_json_env_file, get_project_name

console = Console()


class DeployCommand(BaseCommand, PlusAPIMixin):
    """
    A class to handle deployment-related operations for CrewAI projects.
    """

    def __init__(self):
        """
        Initialize the DeployCommand with project name and API client.
        """

        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)
        self.project_name = get_project_name(require=True)

    def _standard_no_param_error_message(self) -> None:
        """
        Display a standard error message when no UUID or project name is available.
        """
        console.print(
            "No UUID provided, project pyproject.toml not found or with error.",
            style="bold red",
        )

    def _display_deployment_info(self, json_response: Dict[str, Any]) -> None:
        """
        Display deployment information.

        Args:
            json_response (Dict[str, Any]): The deployment information to display.
        """
        console.print("Deploying the crew...\n", style="bold blue")
        for key, value in json_response.items():
            console.print(f"{key.title()}: [green]{value}[/green]")
        console.print("\nTo check the status of the deployment, run:")
        console.print("crewai deploy status")
        console.print(" or")
        console.print(f"crewai deploy status --uuid \"{json_response['uuid']}\"")

    def _display_logs(self, log_messages: List[Dict[str, Any]]) -> None:
        """
        Display log messages.

        Args:
            log_messages (List[Dict[str, Any]]): The log messages to display.
        """
        for log_message in log_messages:
            console.print(
                f"{log_message['timestamp']} - {log_message['level']}: {log_message['message']}"
            )

    def deploy(self, uuid: Optional[str] = None) -> None:
        """
        Deploy a crew using either UUID or project name.

        Args:
            uuid (Optional[str]): The UUID of the crew to deploy.
        """
        self._start_deployment_span = self._telemetry.start_deployment_span(uuid)
        console.print("Starting deployment...", style="bold blue")
        if uuid:
            response = self.plus_api_client.deploy_by_uuid(uuid)
        elif self.project_name:
            response = self.plus_api_client.deploy_by_name(self.project_name)
        else:
            self._standard_no_param_error_message()
            return

        self._validate_response(response)
        self._display_deployment_info(response.json())

    def create_crew(self, confirm: bool = False) -> None:
        """
        Create a new crew deployment.
        """
        self._create_crew_deployment_span = (
            self._telemetry.create_crew_deployment_span()
        )
        console.print("Creating deployment...", style="bold blue")
        env_vars = fetch_and_json_env_file()

        try:
            remote_repo_url = git.Repository().origin_url()
        except ValueError:
            remote_repo_url = None

        if remote_repo_url is None:
            console.print("No remote repository URL found.", style="bold red")
            console.print(
                "Please ensure your project has a valid remote repository.",
                style="yellow",
            )
            return

        self._confirm_input(env_vars, remote_repo_url, confirm)
        payload = self._create_payload(env_vars, remote_repo_url)
        response = self.plus_api_client.create_crew(payload)

        self._validate_response(response)
        self._display_creation_success(response.json())

    def _confirm_input(
        self, env_vars: Dict[str, str], remote_repo_url: str, confirm: bool
    ) -> None:
        """
        Confirm input parameters with the user.

        Args:
            env_vars (Dict[str, str]): Environment variables.
            remote_repo_url (str): Remote repository URL.
            confirm (bool): Whether to confirm input.
        """
        if not confirm:
            input(f"Press Enter to continue with the following Env vars: {env_vars}")
            input(
                f"Press Enter to continue with the following remote repository: {remote_repo_url}\n"
            )

    def _create_payload(
        self,
        env_vars: Dict[str, str],
        remote_repo_url: str,
    ) -> Dict[str, Any]:
        """
        Create the payload for crew creation.

        Args:
            remote_repo_url (str): Remote repository URL.
            env_vars (Dict[str, str]): Environment variables.

        Returns:
            Dict[str, Any]: The payload for crew creation.
        """
        return {
            "deploy": {
                "name": self.project_name,
                "repo_clone_url": remote_repo_url,
                "env": env_vars,
            }
        }

    def _display_creation_success(self, json_response: Dict[str, Any]) -> None:
        """
        Display success message after crew creation.

        Args:
            json_response (Dict[str, Any]): The response containing crew information.
        """
        console.print("Deployment created successfully!\n", style="bold green")
        console.print(
            f"Name: {self.project_name} ({json_response['uuid']})", style="bold green"
        )
        console.print(f"Status: {json_response['status']}", style="bold green")
        console.print("\nTo (re)deploy the crew, run:")
        console.print("crewai deploy push")
        console.print(" or")
        console.print(f"crewai deploy push --uuid {json_response['uuid']}")

    def list_crews(self) -> None:
        """
        List all available crews.
        """
        console.print("Listing all Crews\n", style="bold blue")

        response = self.plus_api_client.list_crews()
        json_response = response.json()
        if response.status_code == 200:
            self._display_crews(json_response)
        else:
            self._display_no_crews_message()

    def _display_crews(self, crews_data: List[Dict[str, Any]]) -> None:
        """
        Display the list of crews.

        Args:
            crews_data (List[Dict[str, Any]]): List of crew data to display.
        """
        for crew_data in crews_data:
            console.print(
                f"- {crew_data['name']} ({crew_data['uuid']}) [blue]{crew_data['status']}[/blue]"
            )

    def _display_no_crews_message(self) -> None:
        """
        Display a message when no crews are available.
        """
        console.print("You don't have any Crews yet. Let's create one!", style="yellow")
        console.print("  crewai create crew <crew_name>", style="green")

    def get_crew_status(self, uuid: Optional[str] = None) -> None:
        """
        Get the status of a crew.

        Args:
            uuid (Optional[str]): The UUID of the crew to check.
        """
        console.print("Fetching deployment status...", style="bold blue")
        if uuid:
            response = self.plus_api_client.crew_status_by_uuid(uuid)
        elif self.project_name:
            response = self.plus_api_client.crew_status_by_name(self.project_name)
        else:
            self._standard_no_param_error_message()
            return

        self._validate_response(response)
        self._display_crew_status(response.json())

    def _display_crew_status(self, status_data: Dict[str, str]) -> None:
        """
        Display the status of a crew.

        Args:
            status_data (Dict[str, str]): The status data to display.
        """
        console.print(f"Name:\t {status_data['name']}")
        console.print(f"Status:\t {status_data['status']}")

    def get_crew_logs(self, uuid: Optional[str], log_type: str = "deployment") -> None:
        """
        Get logs for a crew.

        Args:
            uuid (Optional[str]): The UUID of the crew to get logs for.
            log_type (str): The type of logs to retrieve (default: "deployment").
        """
        self._get_crew_logs_span = self._telemetry.get_crew_logs_span(uuid, log_type)
        console.print(f"Fetching {log_type} logs...", style="bold blue")

        if uuid:
            response = self.plus_api_client.crew_by_uuid(uuid, log_type)
        elif self.project_name:
            response = self.plus_api_client.crew_by_name(self.project_name, log_type)
        else:
            self._standard_no_param_error_message()
            return

        self._validate_response(response)
        self._display_logs(response.json())

    def remove_crew(self, uuid: Optional[str]) -> None:
        """
        Remove a crew deployment.

        Args:
            uuid (Optional[str]): The UUID of the crew to remove.
        """
        self._remove_crew_span = self._telemetry.remove_crew_span(uuid)
        console.print("Removing deployment...", style="bold blue")

        if uuid:
            response = self.plus_api_client.delete_crew_by_uuid(uuid)
        elif self.project_name:
            response = self.plus_api_client.delete_crew_by_name(self.project_name)
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
