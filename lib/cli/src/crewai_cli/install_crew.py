from pathlib import Path
import subprocess

import click
from crewai_core.project import configured_project_definition, read_toml

from crewai_cli.deploy.validate import normalize_package_name
from crewai_cli.utils import build_env_with_all_tool_credentials


def _is_json_crew_project(project_root: Path | None = None) -> bool:
    """Return True for JSON crew projects that do not need package install."""
    root = project_root or Path.cwd()
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        return False

    pyproject = read_toml(pyproject_path)

    if (
        configured_project_definition(
            "crew", pyproject_data=pyproject, project_root=root
        )
        is None
    ):
        return False

    project_name = pyproject.get("project", {}).get("name", "")
    package_name = normalize_package_name(project_name)
    if package_name and (root / "src" / package_name / "crew.py").is_file():
        return False

    return True


# Be mindful about changing this.
# on some environments we don't use this command but instead uv sync directly
# so if you expect this to support more things you will need to replicate it there
# ask @joaomdmoura if you are unsure
def install_crew(
    proxy_options: list[str],
    *,
    raise_on_error: bool = False,
    install_project: bool | None = None,
) -> None:
    """
    Install the crew by running the UV command to lock and install.
    """
    try:
        if install_project is None:
            install_project = not _is_json_crew_project()

        command = ["uv", "sync"]
        if not install_project and "--no-install-project" not in proxy_options:
            command.append("--no-install-project")
        command.extend(proxy_options)

        # Inject tool repository credentials so uv can authenticate
        # against private package indexes (e.g. crewai tool repository).
        # Without this, `uv sync` fails with 401 Unauthorized when the
        # project depends on tools from a private index.
        env = build_env_with_all_tool_credentials()

        subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=False,
            text=True,
            env=env,
        )

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the crew: {e}", err=True)
        click.echo(e.output, err=True)
        if raise_on_error:
            raise

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        if raise_on_error:
            raise
