from typing import Any

from crewai_core.plus_api import CreateCrewPayload
from rich.console import Console

from crewai_cli import git
from crewai_cli.command import BaseCommand, PlusAPIMixin
from crewai_cli.deploy.archive import create_project_zip
from crewai_cli.deploy.validate import validate_project
from crewai_cli.utils import fetch_and_json_env_file, get_project_name


console = Console()


def _run_predeploy_validation(skip_validate: bool) -> bool:
    """Run pre-deploy validation unless skipped.

    Returns True if deployment should proceed, False if it should abort.
    """
    if skip_validate:
        console.print(
            "[yellow]Skipping pre-deploy validation (--skip-validate).[/yellow]"
        )
        return True

    console.print("Running pre-deploy validation...", style="bold blue")
    validator = validate_project()
    if not validator.ok:
        console.print(
            "\n[bold red]Pre-deploy validation failed. "
            "Fix the issues above or re-run with --skip-validate.[/bold red]"
        )
        return False
    return True


def _display_git_repository_help() -> None:
    """Explain how to prepare a new project for deployment."""
    console.print(
        "Initialized a local Git repository and created an initial commit.",
        style="green",
    )


def _display_git_remote_help() -> None:
    """Explain that ZIP deployment will be used without an origin remote."""
    console.print(
        "No origin remote found. Deploying from a ZIP upload instead.",
        style="yellow",
    )


def _env_summary(env_vars: dict[str, str]) -> str:
    if not env_vars:
        return "0 env vars"
    keys = ", ".join(sorted(env_vars))
    return f"{len(env_vars)} env vars: {keys}"


class DeployCommand(BaseCommand, PlusAPIMixin):
    """
    A class to handle deployment-related operations for CrewAI projects.
    """

    def __init__(self) -> None:
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

    def _display_deployment_info(self, json_response: dict[str, Any]) -> None:
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
        console.print(f'crewai deploy status --uuid "{json_response["uuid"]}"')

    def _display_logs(self, log_messages: list[dict[str, Any]]) -> None:
        """
        Display log messages.

        Args:
            log_messages (List[Dict[str, Any]]): The log messages to display.
        """
        for log_message in log_messages:
            console.print(
                f"{log_message['timestamp']} - {log_message['level']}: {log_message['message']}"
            )

    def deploy(self, uuid: str | None = None, skip_validate: bool = False) -> None:
        """
        Deploy a crew using either UUID or project name.

        Args:
            uuid (Optional[str]): The UUID of the crew to deploy.
            skip_validate (bool): Skip pre-deploy validation checks.
        """
        if not _run_predeploy_validation(skip_validate):
            return
        self._telemetry.start_deployment_span(uuid)
        console.print("Starting deployment...", style="bold blue")
        repository = self._prepare_git_repository()
        remote_repo_url = repository.origin_url() if repository else None

        if remote_repo_url and uuid:
            response = self.plus_api_client.deploy_by_uuid(uuid)
        elif remote_repo_url and self.project_name:
            response = self.plus_api_client.deploy_by_name(self.project_name)
        elif uuid:
            _display_git_remote_help()
            response = self._update_crew_from_zip(uuid, repository)
        elif self.project_name:
            _display_git_remote_help()
            deployment_uuid = self._deployment_uuid_by_name()
            response = self._update_crew_from_zip(deployment_uuid, repository)
        else:
            self._standard_no_param_error_message()
            return

        self._validate_response(response)
        self._display_deployment_info(response.json())

    def _deployment_uuid_by_name(self) -> str:
        if not self.project_name:
            raise ValueError("project_name is required to find a deployment")

        response = self.plus_api_client.crew_status_by_name(self.project_name)
        self._validate_response(response)
        json_response = response.json()
        uuid = json_response.get("uuid")
        if not uuid:
            raise ValueError("Deployment status response did not include a uuid")
        return str(uuid)

    def create_crew(self, confirm: bool = False, skip_validate: bool = False) -> None:
        """
        Create a new crew deployment.

        Args:
            confirm (bool): Whether to skip the interactive confirmation prompt.
            skip_validate (bool): Skip pre-deploy validation checks.
        """
        if not _run_predeploy_validation(skip_validate):
            return
        self._telemetry.create_crew_deployment_span()
        console.print("Creating deployment...", style="bold blue")
        env_vars = fetch_and_json_env_file()
        repository = self._prepare_git_repository()
        remote_repo_url = repository.origin_url() if repository else None

        if remote_repo_url:
            self._confirm_input(env_vars, remote_repo_url, confirm)
            payload = self._create_payload(env_vars, remote_repo_url)
            response = self.plus_api_client.create_crew(payload)
        else:
            _display_git_remote_help()
            response = self._create_crew_from_zip(env_vars, repository, confirm)

        self._validate_response(response)
        self._display_creation_success(response.json())

    def _prepare_git_repository(self) -> git.Repository | None:
        try:
            repository = git.Repository()
        except ValueError as exc:
            if "not a Git repository" not in str(exc):
                console.print(
                    f"{exc} Continuing with ZIP deployment.",
                    style="yellow",
                )
                return None

            try:
                repository = git.Repository.initialize()
            except Exception as init_error:
                console.print(
                    "Git auto-setup did not complete. Continuing with ZIP deployment.",
                    style="yellow",
                )
                console.print(str(init_error), style="dim")
                return None

            _display_git_repository_help()
            return repository

        try:
            if repository.create_initial_commit_if_needed():
                console.print(
                    "Created an initial Git commit for this project.",
                    style="green",
                )
        except Exception as commit_error:
            console.print(
                "Could not create an initial Git commit. Continuing with ZIP deployment.",
                style="yellow",
            )
            console.print(str(commit_error), style="dim")
            return None

        return repository

    def _create_crew_from_zip(
        self,
        env_vars: dict[str, str],
        repository: git.Repository | None,
        confirm: bool,
    ) -> Any:
        if not self.project_name:
            raise ValueError("project_name is required to create a ZIP deployment")

        console.print("Preparing project ZIP...", style="bold blue")
        zip_file_path = create_project_zip(self.project_name, repository=repository)
        try:
            self._confirm_zip_input(env_vars, confirm)
            console.print("Uploading project ZIP...", style="bold blue")
            return self.plus_api_client.create_crew_from_zip(
                zip_file_path,
                name=self.project_name,
                env=env_vars,
            )
        finally:
            zip_file_path.unlink(missing_ok=True)

    def _update_crew_from_zip(
        self,
        uuid: str,
        repository: git.Repository | None,
    ) -> Any:
        if not self.project_name:
            raise ValueError("project_name is required to update a ZIP deployment")

        console.print("Preparing project ZIP...", style="bold blue")
        zip_file_path = create_project_zip(self.project_name, repository=repository)
        try:
            console.print("Uploading project ZIP...", style="bold blue")
            return self.plus_api_client.update_crew_from_zip(uuid, zip_file_path)
        finally:
            zip_file_path.unlink(missing_ok=True)

    def _confirm_input(
        self, env_vars: dict[str, str], remote_repo_url: str, confirm: bool
    ) -> None:
        """
        Confirm input parameters with the user.

        Args:
            env_vars (Dict[str, str]): Environment variables.
            remote_repo_url (str): Remote repository URL.
            confirm (bool): Whether to confirm input.
        """
        if not confirm:
            input(f"Press Enter to continue with {_env_summary(env_vars)}")
            input(
                f"Press Enter to continue with the following remote repository: {remote_repo_url}\n"
            )

    def _confirm_zip_input(self, env_vars: dict[str, str], confirm: bool) -> None:
        if not confirm:
            input(f"Press Enter to continue with {_env_summary(env_vars)}")

    def _create_payload(
        self,
        env_vars: dict[str, str],
        remote_repo_url: str,
    ) -> CreateCrewPayload:
        """
        Create the payload for crew creation.

        Args:
            remote_repo_url (str): Remote repository URL.
            env_vars (Dict[str, str]): Environment variables.

        Returns:
            Dict[str, Any]: The payload for crew creation.
        """
        if not self.project_name:
            raise ValueError("project_name is required to create a deployment payload")
        return {
            "deploy": {
                "name": self.project_name,
                "repo_clone_url": remote_repo_url,
                "env": env_vars,
            }
        }

    def _display_creation_success(self, json_response: dict[str, Any]) -> None:
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

    def _display_crews(self, crews_data: list[dict[str, Any]]) -> None:
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

    def get_crew_status(self, uuid: str | None = None) -> None:
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

    def _display_crew_status(self, status_data: dict[str, str]) -> None:
        """
        Display the status of a crew.

        Args:
            status_data (Dict[str, str]): The status data to display.
        """
        console.print(f"Name:\t {status_data['name']}")
        console.print(f"Status:\t {status_data['status']}")

    def get_crew_logs(self, uuid: str | None, log_type: str = "deployment") -> None:
        """
        Get logs for a crew.

        Args:
            uuid (Optional[str]): The UUID of the crew to get logs for.
            log_type (str): The type of logs to retrieve (default: "deployment").
        """
        self._telemetry.get_crew_logs_span(uuid, log_type)
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

    def remove_crew(self, uuid: str | None) -> None:
        """
        Remove a crew deployment.

        Args:
            uuid (Optional[str]): The UUID of the crew to remove.
        """
        self._telemetry.remove_crew_span(uuid)
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
