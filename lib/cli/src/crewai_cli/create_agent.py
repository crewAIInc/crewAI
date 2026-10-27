import keyword
from pathlib import Path

import click
from crewai_core.telemetry import Telemetry

from crewai_cli.git import initialize_if_git_available
from crewai_cli.utils import get_or_create_project_id, normalize_package_name
from crewai_cli.version import get_crewai_dependency_range


def create_agent(name: str) -> None:
    """Create an installable project exporting one standalone Agent."""
    package_name = normalize_package_name(name)
    if not package_name.isidentifier() or keyword.iskeyword(package_name):
        raise click.BadParameter(
            "Use a name that can be a Python package.", param_hint="name"
        )
    project_root = Path(package_name)
    if project_root.exists():
        raise click.ClickException(f"Folder {package_name} already exists.")

    templates_dir = Path(__file__).parent / "templates" / "agent"
    for template in templates_dir.rglob("*"):
        if not template.is_file() or template.suffix in {".pyc", ".pyo"}:
            continue
        relative = template.relative_to(templates_dir)
        destination = project_root / str(relative).replace(
            "{{package_name}}", package_name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = template.read_text(encoding="utf-8")
        content = content.replace("{{package_name}}", package_name)
        content = content.replace(
            "{{crewai_dependency}}", f"crewai{get_crewai_dependency_range()}"
        )
        destination.write_text(content, encoding="utf-8")

    project_id = get_or_create_project_id(project_root / "pyproject.toml")
    Telemetry().project_created_span("agent", project_id)
    initialize_if_git_available(project_root)
    click.secho(f"Agent {package_name} created successfully!", fg="green", bold=True)
