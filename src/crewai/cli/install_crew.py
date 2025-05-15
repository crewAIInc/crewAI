import logging
import subprocess
from typing import List

import click

UV_COMMAND = "uv"
SYNC_COMMAND = "sync"
ACTIVE_FLAG = "--active"

logger = logging.getLogger(__name__)

def install_crew(proxy_options: List[str]) -> None:
    """
    Install the crew by running the UV command to lock and install.
    
    Args:
        proxy_options (List[str]): List of proxy-related command options.
    
    Note:
        Uses --active flag to ensure proper virtual environment detection.
    """
    if not isinstance(proxy_options, list):
        raise ValueError("proxy_options must be a list")

    try:
        command = [UV_COMMAND, SYNC_COMMAND, ACTIVE_FLAG] + proxy_options
        logger.debug(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=False, text=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the crew: {e}", err=True)
        if e.output is not None:
            click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
