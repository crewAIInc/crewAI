import platform
import subprocess
from typing import Any


def run_command(
    command: list[str],
    capture_output: bool = False,
    text: bool = True,
    check: bool = True,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    **kwargs: Any
) -> subprocess.CompletedProcess:
    """
    Cross-platform subprocess execution with Windows compatibility.

    On Windows, uses shell=True to avoid permission issues with restrictive
    security policies. On other platforms, uses the standard approach.

    Args:
        command: List of command arguments
        capture_output: Whether to capture stdout/stderr
        text: Whether to use text mode
        check: Whether to raise CalledProcessError on non-zero exit
        cwd: Working directory
        env: Environment variables
        **kwargs: Additional subprocess.run arguments

    Returns:
        CompletedProcess instance

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
    """
    if platform.system() == "Windows":
        if isinstance(command, list):
            command_str = subprocess.list2cmdline(command)
        else:
            command_str = command

        return subprocess.run(
            command_str,
            shell=True,
            capture_output=capture_output,
            text=text,
            check=check,
            cwd=cwd,
            env=env,
            **kwargs
        )
    return subprocess.run(
        command,
        capture_output=capture_output,
        text=text,
        check=check,
        cwd=cwd,
        env=env,
        **kwargs
    )
