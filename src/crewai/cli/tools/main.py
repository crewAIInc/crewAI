import base64
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from crewai.cli import git
from crewai.cli.command import BaseCommand, PlusAPIMixin
from crewai.cli.config import Settings
from crewai.cli.utils import (
    build_env_with_tool_repository_credentials,
    extract_available_exports,
    get_project_description,
    get_project_name,
    get_project_version,
    tree_copy,
    tree_find_and_replace,
)

console = Console()


class ToolCommand(BaseCommand, PlusAPIMixin):
    """
    A class to handle tool repository related operations for CrewAI projects.
    """

    def __init__(self):
        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)

    def create(self, handle: str):
        self._ensure_not_in_project()

        folder_name = handle.replace(" ", "_").replace("-", "_").lower()
        class_name = handle.replace("_", " ").replace("-", " ").title().replace(" ", "")

        project_root = Path(folder_name)
        if project_root.exists():
            click.secho(f"Folder {folder_name} already exists.", fg="red")
            raise SystemExit
        os.makedirs(project_root)

        click.secho(f"Creating custom tool {folder_name}...", fg="green", bold=True)

        template_dir = Path(__file__).parent.parent / "templates" / "tool"
        tree_copy(template_dir, project_root)
        tree_find_and_replace(project_root, "{{folder_name}}", folder_name)
        tree_find_and_replace(project_root, "{{class_name}}", class_name)

        old_directory = os.getcwd()
        os.chdir(project_root)
        try:
            self.login()
            subprocess.run(["git", "init"], check=True)  # noqa: S607
            console.print(
                f"[green]Created custom tool [bold]{folder_name}[/bold]. Run [bold]cd {project_root}[/bold] to start working.[/green]"
            )
        finally:
            os.chdir(old_directory)

    def publish(self, is_public: bool, force: bool = False):
        if not git.Repository().is_synced() and not force:
            console.print(
                "[bold red]Failed to publish tool.[/bold red]\n"
                "Local changes need to be resolved before publishing. Please do the following:\n"
                "* [bold]Commit[/bold] your changes.\n"
                "* [bold]Push[/bold] to sync with the remote.\n"
                "* [bold]Pull[/bold] the latest changes from the remote.\n"
                "\nOnce your repository is up-to-date, retry publishing the tool."
            )
            raise SystemExit()

        project_name = get_project_name(require=True)
        assert isinstance(project_name, str)  # noqa: S101

        project_version = get_project_version(require=True)
        assert isinstance(project_version, str)  # noqa: S101

        project_description = get_project_description(require=False)
        encoded_tarball = None

        console.print("[bold blue]Discovering tools from your project...[/bold blue]")
        available_exports = extract_available_exports()

        if available_exports:
            console.print(
                f"[green]Found these tools to publish: {', '.join([e['name'] for e in available_exports])}[/green]"
            )
        self._print_current_organization()

        with tempfile.TemporaryDirectory() as temp_build_dir:
            subprocess.run(  # noqa: S603
                ["uv", "build", "--sdist", "--out-dir", temp_build_dir],  # noqa: S607
                check=True,
                capture_output=False,
            )

            tarball_filename = next(
                (f for f in os.listdir(temp_build_dir) if f.endswith(".tar.gz")), None
            )
            if not tarball_filename:
                console.print(
                    "Project build failed. Please ensure that the command `uv build --sdist` completes successfully.",
                    style="bold red",
                )
                raise SystemExit

            tarball_path = os.path.join(temp_build_dir, tarball_filename)
            with open(tarball_path, "rb") as file:
                tarball_contents = file.read()

            encoded_tarball = base64.b64encode(tarball_contents).decode("utf-8")

        console.print("[bold blue]Publishing tool to repository...[/bold blue]")
        publish_response = self.plus_api_client.publish_tool(
            handle=project_name,
            is_public=is_public,
            version=project_version,
            description=project_description,
            encoded_file=f"data:application/x-gzip;base64,{encoded_tarball}",
            available_exports=available_exports,
        )

        self._validate_response(publish_response)

        published_handle = publish_response.json()["handle"]
        console.print(
            f"Successfully published `{published_handle}` ({project_version}).\n\n"
            + "⚠️ Security checks are running in the background. Your tool will be available once these are complete.\n"
            + f"You can monitor the status or access your tool here:\nhttps://app.crewai.com/crewai_plus/tools/{published_handle}",
            style="bold green",
        )

    def install(self, handle: str):
        self._print_current_organization()
        get_response = self.plus_api_client.get_tool(handle)

        if get_response.status_code == 404:
            console.print(
                "No tool found with this name. Please ensure the tool was published and you have access to it.",
                style="bold red",
            )
            raise SystemExit
        if get_response.status_code != 200:
            console.print(
                "Failed to get tool details. Please try again later.", style="bold red"
            )
            raise SystemExit

        self._add_package(get_response.json())

        console.print(f"Successfully installed {handle}", style="bold green")

    def login(self) -> None:
        login_response = self.plus_api_client.login_to_tool_repository()

        if login_response.status_code != 200:
            console.print(
                "Authentication failed. Verify access to the tool repository, or try `crewai login`. ",
                style="bold red",
            )
            raise SystemExit

        login_response_json = login_response.json()

        settings = Settings()
        settings.tool_repository_username = login_response_json["credential"][
            "username"
        ]
        settings.tool_repository_password = login_response_json["credential"][
            "password"
        ]
        settings.org_uuid = login_response_json["current_organization"]["uuid"]
        settings.org_name = login_response_json["current_organization"]["name"]
        settings.dump()

    def _add_package(self, tool_details: dict[str, Any]):
        is_from_pypi = tool_details.get("source", None) == "pypi"
        tool_handle = tool_details["handle"]
        repository_handle = tool_details["repository"]["handle"]
        repository_url = tool_details["repository"]["url"]
        index = f"{repository_handle}={repository_url}"

        add_package_command = [
            "uv",
            "add",
        ]

        if is_from_pypi:
            add_package_command.append(tool_handle)
        else:
            add_package_command.extend(["--index", index, tool_handle])

        add_package_result = subprocess.run(  # noqa: S603
            add_package_command,
            capture_output=False,
            env=build_env_with_tool_repository_credentials(repository_handle),
            text=True,
            check=True,
        )

        if add_package_result.stderr:
            click.echo(add_package_result.stderr, err=True)
            raise SystemExit

    def _ensure_not_in_project(self):
        if os.path.isfile("./pyproject.toml"):
            console.print(
                "[bold red]Oops! It looks like you're inside a project.[/bold red]"
            )
            console.print(
                "You can't create a new tool while inside an existing project."
            )
            console.print(
                "[bold yellow]Tip:[/bold yellow] Navigate to a different directory and try again."
            )
            raise SystemExit

    def _print_current_organization(self) -> None:
        settings = Settings()
        if settings.org_uuid:
            console.print(
                f"Current organization: {settings.org_name} ({settings.org_uuid})",
                style="bold blue",
            )
        else:
            console.print(
                "No organization currently set. We recommend setting one before using: `crewai org switch <org_id>` command.",
                style="yellow",
            )
