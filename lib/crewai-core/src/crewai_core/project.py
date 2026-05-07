"""TOML / pyproject.toml utilities shared by crewai and crewai-cli."""

from __future__ import annotations

from functools import reduce
import sys
from typing import Any

from rich.console import Console
import tomli


if sys.version_info >= (3, 11):
    import tomllib

console = Console()


def read_toml(file_path: str = "pyproject.toml") -> dict[str, Any]:
    """Read a TOML file from disk and return its parsed contents."""
    with open(file_path, "rb") as f:
        return tomli.load(f)


def parse_toml(content: str) -> dict[str, Any]:
    """Parse a TOML string and return its parsed contents."""
    if sys.version_info >= (3, 11):
        return tomllib.loads(content)
    return tomli.loads(content)


def _get_nested_value(data: dict[str, Any], keys: list[str]) -> Any:
    return reduce(dict.__getitem__, keys, data)


def _get_project_attribute(
    pyproject_path: str, keys: list[str], require: bool
) -> Any | None:
    """Look up a dotted attribute path inside ``pyproject_path``.

    The file must declare ``crewai`` in ``[project].dependencies`` for the
    lookup to succeed (a guard against running these helpers outside a crewai
    project directory). When ``require=True``, missing attributes raise
    ``SystemExit`` after printing a friendly error.
    """
    attribute = None

    try:
        with open(pyproject_path, "r") as f:
            pyproject_content = parse_toml(f.read())

        dependencies = (
            _get_nested_value(pyproject_content, ["project", "dependencies"]) or []
        )
        if not any(True for dep in dependencies if "crewai" in dep):
            raise Exception("crewai is not in the dependencies.")

        attribute = _get_nested_value(pyproject_content, keys)
    except FileNotFoundError:
        console.print(f"Error: {pyproject_path} not found.", style="bold red")
    except KeyError:
        console.print(
            f"Error: {pyproject_path} is not a valid pyproject.toml file.",
            style="bold red",
        )
    except Exception as e:
        if sys.version_info >= (3, 11) and isinstance(e, tomllib.TOMLDecodeError):
            console.print(
                f"Error: {pyproject_path} is not a valid TOML file.", style="bold red"
            )
        else:
            console.print(
                f"Error reading the pyproject.toml file: {e}", style="bold red"
            )

    if require and not attribute:
        console.print(
            f"Unable to read '{'.'.join(keys)}' in the pyproject.toml file. "
            "Please verify that the file exists and contains the specified attribute.",
            style="bold red",
        )
        raise SystemExit

    return attribute


def get_project_name(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Return the project name from ``pyproject.toml``."""
    return _get_project_attribute(pyproject_path, ["project", "name"], require=require)


def get_project_version(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Return the project version from ``pyproject.toml``."""
    return _get_project_attribute(
        pyproject_path, ["project", "version"], require=require
    )


def get_project_description(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Return the project description from ``pyproject.toml``."""
    return _get_project_attribute(
        pyproject_path, ["project", "description"], require=require
    )
