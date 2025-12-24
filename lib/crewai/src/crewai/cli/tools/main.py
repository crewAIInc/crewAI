import base64
from json import JSONDecodeError
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import click
from rich.console import Console

from crewai.cli import git
from crewai.cli.command import BaseCommand, PlusAPIMixin
from crewai.cli.config import Settings
from crewai.cli.constants import DEFAULT_CREWAI_ENTERPRISE_URL
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

    def __init__(self) -> None:
        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)

    def create(self, handle: str) -> None:
        self._ensure_not_in_project()

        folder_name = handle.replace(" ", "_").replace("-", "_").lower()
        class_name = handle.replace("_", " ").replace("-", " ").title().replace(" ", "")

        project_root = Path(folder_name)
        if project_root.exists():
            click.secho(f"Map {folder_name} bestaat al.", fg="red")
            raise SystemExit
        os.makedirs(project_root)

        click.secho(f"Custom tool {folder_name} wordt aangemaakt...", fg="green", bold=True)

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
                f"[green]Custom tool [bold]{folder_name}[/bold] aangemaakt. Voer [bold]cd {project_root}[/bold] uit om te beginnen.[/green]"
            )
        finally:
            os.chdir(old_directory)

    def publish(self, is_public: bool, force: bool = False) -> None:
        if not git.Repository().is_synced() and not force:
            console.print(
                "[bold red]Tool publiceren mislukt.[/bold red]\n"
                "Lokale wijzigingen moeten worden opgelost voor publicatie. Doe het volgende:\n"
                "* [bold]Commit[/bold] je wijzigingen.\n"
                "* [bold]Push[/bold] om te synchroniseren met de remote.\n"
                "* [bold]Pull[/bold] de laatste wijzigingen van de remote.\n"
                "\nProbeer de tool opnieuw te publiceren zodra je repository up-to-date is."
            )
            raise SystemExit()

        project_name = get_project_name(require=True)
        assert isinstance(project_name, str)  # noqa: S101

        project_version = get_project_version(require=True)
        assert isinstance(project_version, str)  # noqa: S101

        project_description = get_project_description(require=False)
        encoded_tarball = None

        console.print("[bold blue]Tools worden ontdekt in je project...[/bold blue]")
        available_exports = extract_available_exports()

        if available_exports:
            console.print(
                f"[green]Deze tools gevonden om te publiceren: {', '.join([e['name'] for e in available_exports])}[/green]"
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
                    "Project build mislukt. Zorg ervoor dat het commando `uv build --sdist` succesvol wordt uitgevoerd.",
                    style="bold red",
                )
                raise SystemExit

            tarball_path = os.path.join(temp_build_dir, tarball_filename)
            with open(tarball_path, "rb") as file:
                tarball_contents = file.read()

            encoded_tarball = base64.b64encode(tarball_contents).decode("utf-8")

        console.print("[bold blue]Tool wordt gepubliceerd naar repository...[/bold blue]")
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
        settings = Settings()
        base_url = settings.enterprise_base_url or DEFAULT_CREWAI_ENTERPRISE_URL

        console.print(
            f"`{published_handle}` ({project_version}) succesvol gepubliceerd.\n\n"
            + "⚠️ Beveiligingscontroles worden op de achtergrond uitgevoerd. Je tool is beschikbaar zodra deze zijn voltooid.\n"
            + f"Je kunt de status monitoren of je tool hier openen:\n{base_url}/crewai_plus/tools/{published_handle}",
            style="bold green",
        )

    def install(self, handle: str) -> None:
        self._print_current_organization()
        get_response = self.plus_api_client.get_tool(handle)

        if get_response.status_code == 404:
            console.print(
                "Geen tool gevonden met deze naam. Zorg ervoor dat de tool is gepubliceerd en dat je er toegang toe hebt.",
                style="bold red",
            )
            raise SystemExit
        if get_response.status_code != 200:
            console.print(
                "Tool details ophalen mislukt. Probeer het later opnieuw.", style="bold red"
            )
            raise SystemExit

        self._add_package(get_response.json())

        console.print(f"{handle} succesvol geïnstalleerd", style="bold green")

    def login(self) -> None:
        login_response = self.plus_api_client.login_to_tool_repository()

        if login_response.status_code != 200:
            console.print(
                "Authenticatie mislukt. Controleer of de huidige actieve organisatie toegang heeft tot de tool repository, en voer 'crewai login' opnieuw uit.",
                style="bold red",
            )
            try:
                console.print(
                    f"[{login_response.status_code} error - {login_response.json().get('message', 'Unknown error')}]",
                    style="bold red italic",
                )
            except JSONDecodeError:
                console.print(
                    f"[{login_response.status_code} error - Unknown error - Invalid JSON response]",
                    style="bold red italic",
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

    def _add_package(self, tool_details: dict[str, Any]) -> None:
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

    def _ensure_not_in_project(self) -> None:
        if os.path.isfile("./pyproject.toml"):
            console.print(
                "[bold red]Oeps! Het lijkt erop dat je in een project zit.[/bold red]"
            )
            console.print(
                "Je kunt geen nieuwe tool aanmaken terwijl je in een bestaand project zit."
            )
            console.print(
                "[bold yellow]Tip:[/bold yellow] Navigeer naar een andere directory en probeer opnieuw."
            )
            raise SystemExit

    def _print_current_organization(self) -> None:
        settings = Settings()
        if settings.org_uuid:
            console.print(
                f"Huidige organisatie: {settings.org_name} ({settings.org_uuid})",
                style="bold blue",
            )
        else:
            console.print(
                "Geen organisatie ingesteld. We raden aan om er een in te stellen met: `crewai org switch <org_id>` commando.",
                style="yellow",
            )
