import io
import logging
import os
import shutil
from typing import Any
import zipfile

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from crewai.cli.command import BaseCommand


logger = logging.getLogger(__name__)
console = Console()

GITHUB_ORG = "crewAIInc"
TEMPLATE_PREFIX = "template_"
GITHUB_API_BASE = "https://api.github.com"

BANNER = """\
[bold white] ██████╗██████╗ ███████╗██╗    ██╗[/bold white] [bold red] █████╗ ██╗[/bold red]
[bold white]██╔════╝██╔══██╗██╔════╝██║    ██║[/bold white] [bold red]██╔══██╗██║[/bold red]
[bold white]██║     ██████╔╝█████╗  ██║ █╗ ██║[/bold white] [bold red]███████║██║[/bold red]
[bold white]██║     ██╔══██╗██╔══╝  ██║███╗██║[/bold white] [bold red]██╔══██║██║[/bold red]
[bold white]╚██████╗██║  ██║███████╗╚███╔███╔╝[/bold white] [bold red]██║  ██║██║[/bold red]
[bold white] ╚═════╝╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝[/bold white] [bold red]╚═╝  ╚═╝╚═╝[/bold red]
[dim white]████████╗███████╗███╗   ███╗██████╗ ██╗      █████╗ ████████╗███████╗███████╗[/dim white]
[dim white]╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝██╔════╝[/dim white]
[dim white]   ██║   █████╗  ██╔████╔██║██████╔╝██║     ███████║   ██║   █████╗  ███████╗[/dim white]
[dim white]   ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  ╚════██║[/dim white]
[dim white]   ██║   ███████╗██║ ╚═╝ ██║██║     ███████╗██║  ██║   ██║   ███████╗███████║[/dim white]
[dim white]   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝[/dim white]"""


class TemplateCommand(BaseCommand):
    """Handle template-related operations for CrewAI projects."""

    def __init__(self) -> None:
        super().__init__()

    def list_templates(self) -> None:
        """List available templates with an interactive selector to install."""
        templates = self._fetch_templates()
        if not templates:
            click.echo("No templates found.")
            return

        console.print(f"\n{BANNER}\n")
        console.print(" [on cyan] templates [/on cyan]\n")
        console.print(f" [green]o[/green]  Source: https://github.com/{GITHUB_ORG}")
        console.print(
            f" [green]o[/green]  Found [bold]{len(templates)}[/bold] templates\n"
        )
        console.print(" [green]o[/green]  Select a template to install")

        for idx, repo in enumerate(templates, start=1):
            name = repo["name"].removeprefix(TEMPLATE_PREFIX)
            description = repo.get("description") or ""
            if description:
                console.print(
                    f"      [bold cyan]{idx}.[/bold cyan] [bold white]{name}[/bold white] [dim]({description})[/dim]"
                )
            else:
                console.print(
                    f"      [bold cyan]{idx}.[/bold cyan] [bold white]{name}[/bold white]"
                )

        console.print("      [bold cyan]q.[/bold cyan] [dim]Quit[/dim]\n")

        while True:
            choice = click.prompt("Enter your choice", type=str)

            if choice.lower() == "q":
                return

            if choice.isdigit() and 1 <= int(choice) <= len(templates):
                selected_index = int(choice) - 1
                break

            click.secho(
                f"Please enter a number between 1 and {len(templates)}, or 'q' to quit.",
                fg="yellow",
            )

        selected = templates[selected_index]
        repo_name = selected["name"]
        self._install_repo(repo_name)

    def add_template(self, name: str, output_dir: str | None = None) -> None:
        """Download a template and copy it into the current working directory.

        Args:
            name: Template name (with or without the template_ prefix).
            output_dir: Optional directory name. Defaults to the template name.
        """
        repo_name = self._resolve_repo_name(name)
        if repo_name is None:
            click.secho(f"Template '{name}' not found.", fg="red")
            click.echo("Run 'crewai template list' to see available templates.")
            raise SystemExit(1)

        self._install_repo(repo_name, output_dir)

    def _install_repo(self, repo_name: str, output_dir: str | None = None) -> None:
        """Download and extract a template repo into the current directory.

        Args:
            repo_name: Full GitHub repo name (e.g. template_deep_research).
            output_dir: Optional directory name. Defaults to the template name.
        """
        folder_name = output_dir or repo_name.removeprefix(TEMPLATE_PREFIX)
        dest = os.path.join(os.getcwd(), folder_name)

        while os.path.exists(dest):
            click.secho(f"Directory '{folder_name}' already exists.", fg="yellow")
            folder_name = click.prompt(
                "Enter a different directory name (or 'q' to quit)", type=str
            )
            if folder_name.lower() == "q":
                return
            dest = os.path.join(os.getcwd(), folder_name)

        click.echo(
            f"Downloading template '{repo_name.removeprefix(TEMPLATE_PREFIX)}'..."
        )

        zip_bytes = self._download_zip(repo_name)
        self._extract_zip(zip_bytes, dest)

        self._telemetry.template_installed_span(repo_name.removeprefix(TEMPLATE_PREFIX))

        console.print(
            f"\n [green]\u2713[/green]  Installed template [bold white]{folder_name}[/bold white]"
            f" [dim](source: github.com/{GITHUB_ORG}/{repo_name})[/dim]\n"
        )

        next_steps = Text()
        next_steps.append(f"  cd {folder_name}\n", style="bold white")
        next_steps.append("  crewai install", style="bold white")

        panel = Panel(
            next_steps,
            title="[green]\u25c7  Next steps[/green]",
            title_align="left",
            border_style="dim",
            padding=(1, 2),
        )
        console.print(panel)

    def _fetch_templates(self) -> list[dict[str, Any]]:
        """Fetch all template repos from the GitHub org."""
        templates: list[dict[str, Any]] = []
        page = 1
        while True:
            url = f"{GITHUB_API_BASE}/orgs/{GITHUB_ORG}/repos"
            params: dict[str, str | int] = {
                "per_page": 100,
                "page": page,
                "type": "public",
            }
            try:
                response = httpx.get(url, params=params, timeout=15)
                response.raise_for_status()
            except httpx.HTTPError as e:
                click.secho(f"Failed to fetch templates from GitHub: {e}", fg="red")
                raise SystemExit(1) from e

            repos = response.json()
            if not repos:
                break

            templates.extend(
                repo
                for repo in repos
                if repo["name"].startswith(TEMPLATE_PREFIX) and not repo.get("private")
            )

            page += 1

        templates.sort(key=lambda r: r["name"])
        return templates

    def _resolve_repo_name(self, name: str) -> str | None:
        """Resolve user input to a full repo name, or None if not found."""
        # Accept both 'deep_research' and 'template_deep_research'
        candidates = [
            f"{TEMPLATE_PREFIX}{name}"
            if not name.startswith(TEMPLATE_PREFIX)
            else name,
            name,
        ]

        templates = self._fetch_templates()
        template_names = {t["name"] for t in templates}

        for candidate in candidates:
            if candidate in template_names:
                return candidate

        return None

    def _download_zip(self, repo_name: str) -> bytes:
        """Download the default branch zipball for a repo."""
        url = f"{GITHUB_API_BASE}/repos/{GITHUB_ORG}/{repo_name}/zipball"
        try:
            response = httpx.get(url, follow_redirects=True, timeout=60)
            response.raise_for_status()
        except httpx.HTTPError as e:
            click.secho(f"Failed to download template: {e}", fg="red")
            raise SystemExit(1) from e

        return response.content

    def _extract_zip(self, zip_bytes: bytes, dest: str) -> None:
        """Extract a GitHub zipball into dest, stripping the top-level directory."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # GitHub zipballs have a single top-level dir like 'crewAIInc-template_xxx-<sha>/'
            members = zf.namelist()
            if not members:
                click.secho("Downloaded archive is empty.", fg="red")
                raise SystemExit(1)

            top_dir = members[0].split("/")[0] + "/"

            os.makedirs(dest, exist_ok=True)

            for member in members:
                if member == top_dir or not member.startswith(top_dir):
                    continue

                relative_path = member[len(top_dir) :]
                if not relative_path:
                    continue

                target = os.path.realpath(os.path.join(dest, relative_path))
                if not target.startswith(
                    os.path.realpath(dest) + os.sep
                ) and target != os.path.realpath(dest):
                    continue

                if member.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
