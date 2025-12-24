from pathlib import Path

import click

from crewai.cli.utils import copy_template
from crewai.utilities.printer import Printer


_printer = Printer()


def add_crew_to_flow(crew_name: str) -> None:
    """Add a new crew to the current flow."""
    # Check if pyproject.toml exists in the current directory
    if not Path("pyproject.toml").exists():
        _printer.print(
            "Dit commando moet worden uitgevoerd vanuit de root van een flow project.", color="red"
        )
        raise click.ClickException(
            "Dit commando moet worden uitgevoerd vanuit de root van een flow project."
        )

    # Determine the flow folder based on the current directory
    flow_folder = Path.cwd()
    crews_folder = flow_folder / "src" / flow_folder.name / "crews"

    if not crews_folder.exists():
        _printer.print("Crews map bestaat niet in de huidige flow.", color="red")
        raise click.ClickException("Crews map bestaat niet in de huidige flow.")

    # Create the crew within the flow's crews directory
    create_embedded_crew(crew_name, parent_folder=crews_folder)

    click.echo(
        f"Crew {crew_name} succesvol toegevoegd aan de huidige flow!",
    )


def create_embedded_crew(crew_name: str, parent_folder: Path) -> None:
    """Create a new crew within an existing flow project."""
    folder_name = crew_name.replace(" ", "_").replace("-", "_").lower()
    class_name = crew_name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    crew_folder = parent_folder / folder_name

    if crew_folder.exists():
        if not click.confirm(
            f"Crew {folder_name} bestaat al. Wil je deze overschrijven?"
        ):
            click.secho("Operatie geannuleerd.", fg="yellow")
            return
        click.secho(f"Crew {folder_name} wordt overschreven...", fg="green", bold=True)
    else:
        click.secho(f"Crew {folder_name} wordt aangemaakt...", fg="green", bold=True)
        crew_folder.mkdir(parents=True)

    # Create config and crew.py files
    config_folder = crew_folder / "config"
    config_folder.mkdir(exist_ok=True)

    templates_dir = Path(__file__).parent / "templates" / "crew"
    config_template_files = ["agents.yaml", "tasks.yaml"]
    crew_template_file = f"{folder_name}.py"  # Updated file name

    for file_name in config_template_files:
        src_file = templates_dir / "config" / file_name
        dst_file = config_folder / file_name
        copy_template(src_file, dst_file, crew_name, class_name, folder_name)

    src_file = templates_dir / "crew.py"
    dst_file = crew_folder / crew_template_file
    copy_template(src_file, dst_file, crew_name, class_name, folder_name)

    click.secho(
        f"Crew {crew_name} succesvol toegevoegd aan de flow!", fg="green", bold=True
    )
