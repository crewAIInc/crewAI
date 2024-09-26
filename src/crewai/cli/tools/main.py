import base64
import click
import os
import subprocess
import tempfile

from crewai.cli.command import BaseCommand, PlusAPIMixin
from crewai.cli.utils import (
    get_project_name,
    get_project_description,
    get_project_version,
)
from rich.console import Console

console = Console()


class ToolCommand(BaseCommand, PlusAPIMixin):
    """
    A class to handle tool repository related operations for CrewAI projects.
    """

    def __init__(self):
        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)

    def publish(self, is_public: bool):
        project_name = get_project_name(require=True)
        assert isinstance(project_name, str)

        project_version = get_project_version(require=True)
        assert isinstance(project_version, str)

        project_description = get_project_description(require=False)
        encoded_tarball = None

        with tempfile.TemporaryDirectory() as temp_build_dir:
            subprocess.run(
                ["poetry", "build", "-f", "sdist", "--output", temp_build_dir],
                check=True,
                capture_output=False,
            )

            tarball_filename = next(
                (f for f in os.listdir(temp_build_dir) if f.endswith(".tar.gz")), None
            )
            if not tarball_filename:
                console.print(
                    "Project build failed. Please ensure that the command `poetry build -f sdist` completes successfully.",
                    style="bold red",
                )
                raise SystemExit

            tarball_path = os.path.join(temp_build_dir, tarball_filename)
            with open(tarball_path, "rb") as file:
                tarball_contents = file.read()

            encoded_tarball = base64.b64encode(tarball_contents).decode("utf-8")

        publish_response = self.plus_api_client.publish_tool(
            handle=project_name,
            is_public=is_public,
            version=project_version,
            description=project_description,
            encoded_file=f"data:application/x-gzip;base64,{encoded_tarball}",
        )
        if publish_response.status_code == 422:
            console.print(
                "[bold red]Failed to publish tool. Please fix the following errors:[/bold red]"
            )
            for field, messages in publish_response.json().items():
                for message in messages:
                    console.print(
                        f"* [bold red]{field.capitalize()}[/bold red] {message}"
                    )

            raise SystemExit
        elif publish_response.status_code != 200:
            self._handle_plus_api_error(publish_response.json())
            console.print(
                "Failed to publish tool. Please try again later.", style="bold red"
            )
            raise SystemExit

        published_handle = publish_response.json()["handle"]
        console.print(
            f"Succesfully published {published_handle} ({project_version}).\nInstall it in other projects with crewai tool install {published_handle}",
            style="bold green",
        )

    def install(self, handle: str):
        get_response = self.plus_api_client.get_tool(handle)

        if get_response.status_code == 404:
            console.print(
                "No tool found with this name. Please ensure the tool was published and you have access to it.",
                style="bold red",
            )
            raise SystemExit
        elif get_response.status_code != 200:
            console.print(
                "Failed to get tool details. Please try again later.", style="bold red"
            )
            raise SystemExit

        self._add_repository_to_poetry(get_response.json())
        self._add_package(get_response.json())

        console.print(f"Succesfully installed {handle}", style="bold green")

    def _add_repository_to_poetry(self, tool_details):
        repository_handle = f"crewai-{tool_details['repository']['handle']}"
        repository_url = tool_details["repository"]["url"]
        repository_credentials = tool_details["repository"]["credentials"]

        add_repository_command = [
            "poetry",
            "source",
            "add",
            "--priority=explicit",
            repository_handle,
            repository_url,
        ]
        add_repository_result = subprocess.run(
            add_repository_command, text=True, check=True
        )

        if add_repository_result.stderr:
            click.echo(add_repository_result.stderr, err=True)
            raise SystemExit

        add_repository_credentials_command = [
            "poetry",
            "config",
            f"http-basic.{repository_handle}",
            repository_credentials,
            '""',
        ]
        add_repository_credentials_result = subprocess.run(
            add_repository_credentials_command,
            capture_output=False,
            text=True,
            check=True,
        )

        if add_repository_credentials_result.stderr:
            click.echo(add_repository_credentials_result.stderr, err=True)
            raise SystemExit

    def _add_package(self, tool_details):
        tool_handle = tool_details["handle"]
        repository_handle = tool_details["repository"]["handle"]
        pypi_index_handle = f"crewai-{repository_handle}"

        add_package_command = [
            "poetry",
            "add",
            "--source",
            pypi_index_handle,
            tool_handle,
        ]
        add_package_result = subprocess.run(
            add_package_command, capture_output=False, text=True, check=True
        )

        if add_package_result.stderr:
            click.echo(add_package_result.stderr, err=True)
            raise SystemExit
