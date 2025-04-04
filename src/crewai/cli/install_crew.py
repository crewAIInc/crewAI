import subprocess
from pathlib import Path
from typing import List

import click


def _check_pyproject_exists() -> bool:
    """
    Check if pyproject.toml exists in the current directory.
    
    Returns:
        bool: True if pyproject.toml exists, False otherwise.
    """
    if not Path("pyproject.toml").exists():
        click.echo("Error: No pyproject.toml found in current directory.", err=True)
        click.echo("This command must be run from the root of a crew project.", err=True)
        return False
    return True


def install_crew(proxy_options: List[str]) -> None:
    """
    Install the crew by running the UV command to lock and install.
    
    Args:
        proxy_options: List of proxy options to pass to UV.
    """
    if not _check_pyproject_exists():
        return

    try:
        command = ["uv", "sync"] + proxy_options
        subprocess.run(command, check=True, capture_output=False, text=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the crew: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
