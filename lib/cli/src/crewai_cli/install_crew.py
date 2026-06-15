from pathlib import Path
import subprocess

import click

from crewai_cli.utils import build_env_with_all_tool_credentials, parse_toml


def _find_json_crew_file(project_root: Path | None = None) -> Path | None:
    """Return the JSON crew definition path when present."""
    root = project_root or Path.cwd()
    for filename in ("crew.jsonc", "crew.json"):
        crew_path = root / filename
        if crew_path.is_file():
            return crew_path
    return None


def _is_json_crew_project(project_root: Path | None = None) -> bool:
    """Return True for JSON crew projects that do not need package install."""
    root = project_root or Path.cwd()
    if _find_json_crew_file(root) is None:
        return False

    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        return True

    try:
        pyproject = parse_toml(pyproject_path.read_text())
    except Exception:
        return True

    declared_type: str | None = (
        (pyproject.get("tool") or {}).get("crewai", {}).get("type")
    )
    return declared_type != "flow"


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
