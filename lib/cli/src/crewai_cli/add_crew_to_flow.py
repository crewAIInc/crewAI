from pathlib import Path

import click
from crewai_core.printer import PRINTER
from crewai_core.project import read_toml

from crewai_cli.utils import copy_template, normalize_package_name


def add_crew_to_flow(crew_name: str) -> None:
    """Add a new crew to the current flow."""
    if not Path("pyproject.toml").exists():
        PRINTER.print(
            "This command must be run from the root of a flow project.", color="red"
        )
        raise click.ClickException(
            "This command must be run from the root of a flow project."
        )

    crews_folder = _find_crews_folder(Path.cwd())

    create_embedded_crew(crew_name, parent_folder=crews_folder)

    click.echo(
        f"Crew {crew_name} added to the current flow successfully!",
    )


def _find_crews_folder(flow_folder: Path) -> Path:
    """Locate the flow package's crews folder from ``[project].name``.

    The package directory follows ``[project].name``, not the folder the project
    was cloned into. Only a project without a name falls back to a single
    ``src/*/crews``, so a crew is never written into a package the name contradicts.
    """
    project = read_toml(flow_folder / "pyproject.toml").get("project", {})
    package_name = normalize_package_name(project.get("name", ""))
    if package_name:
        expected = Path("src") / package_name / "crews"
        if (flow_folder / expected).is_dir():
            return flow_folder / expected
        message = (
            f"Crews folder does not exist in the current flow (expected {expected})."
        )
    else:
        found = sorted(
            path for path in flow_folder.glob("src/*/crews") if path.is_dir()
        )
        if len(found) == 1:
            return found[0]
        names = ", ".join(str(path.relative_to(flow_folder)) for path in found)
        message = (
            f"pyproject.toml has no [project].name and found {len(found)} crews "
            f"folders under src/ ({names or 'none'}); expected exactly one."
        )
    PRINTER.print(message, color="red")
    raise click.ClickException(message)


def create_embedded_crew(crew_name: str, parent_folder: Path) -> None:
    """Create a new crew within an existing flow project."""
    folder_name = crew_name.replace(" ", "_").replace("-", "_").lower()
    class_name = crew_name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    crew_folder = parent_folder / folder_name

    if crew_folder.exists():
        if not click.confirm(
            f"Crew {folder_name} already exists. Do you want to override it?"
        ):
            click.secho("Operation cancelled.", fg="yellow")
            return
        click.secho(f"Overriding crew {folder_name}...", fg="green", bold=True)
    else:
        click.secho(f"Creating crew {folder_name}...", fg="green", bold=True)
        crew_folder.mkdir(parents=True)

    config_folder = crew_folder / "config"
    config_folder.mkdir(exist_ok=True)

    templates_dir = Path(__file__).parent / "templates" / "crew"
    config_template_files = ["agents.yaml", "tasks.yaml"]
    crew_template_file = f"{folder_name}.py"

    for file_name in config_template_files:
        src_file = templates_dir / "config" / file_name
        dst_file = config_folder / file_name
        copy_template(src_file, dst_file, crew_name, class_name, folder_name)

    src_file = templates_dir / "crew.py"
    dst_file = crew_folder / crew_template_file
    copy_template(src_file, dst_file, crew_name, class_name, folder_name)

    click.secho(
        f"Crew {crew_name} added to the flow successfully!", fg="green", bold=True
    )
