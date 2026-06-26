"""TOML / pyproject.toml utilities shared by crewai and crewai-cli."""

from __future__ import annotations

from functools import reduce
from pathlib import Path, PureWindowsPath
import sys
from typing import Any

from rich.console import Console
import tomli


if sys.version_info >= (3, 11):
    import tomllib

console = Console()


class ProjectDefinitionError(ValueError):
    """Invalid ``[tool.crewai].definition`` project configuration."""


def read_toml(file_path: str | Path = "pyproject.toml") -> dict[str, Any]:
    """Read a TOML file from disk and return its parsed contents."""
    with open(file_path, "rb") as f:
        return tomli.load(f)


def parse_toml(content: str) -> dict[str, Any]:
    """Parse a TOML string and return its parsed contents."""
    if sys.version_info >= (3, 11):
        return tomllib.loads(content)
    return tomli.loads(content)


def get_crewai_project_config(pyproject_data: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized ``[tool.crewai]`` table from pyproject data."""
    tool_config = pyproject_data.get("tool")
    if not isinstance(tool_config, dict):
        return {}
    crewai_config = tool_config.get("crewai")
    if not isinstance(crewai_config, dict):
        return {}
    return crewai_config


def get_crewai_project_type(pyproject_data: dict[str, Any]) -> str | None:
    """Return ``[tool.crewai].type`` when configured."""
    project_type = get_crewai_project_config(pyproject_data).get("type")
    return project_type if isinstance(project_type, str) else None


def configured_project_definition(
    project_type: str,
    *,
    pyproject_data: dict[str, Any] | None = None,
    project_root: Path | str | None = None,
) -> Path | None:
    """Return a configured CrewAI definition path for a project type.

    ``[tool.crewai].type`` must match ``project_type`` and ``definition`` must
    be a non-empty project-local file path. Missing definitions return ``None``
    so callers can fall back to legacy entrypoints for that project type.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    if pyproject_data is None:
        pyproject_data = read_toml(root / "pyproject.toml")

    crewai_config = get_crewai_project_config(pyproject_data)
    if crewai_config.get("type") != project_type:
        return None

    definition = crewai_config.get("definition", "").strip()
    if not definition:
        return None

    return resolve_project_definition_path(definition=definition, project_root=root)


def resolve_project_definition_path(definition: str, project_root: Path | str) -> Path:
    """Resolve a ``[tool.crewai].definition`` path inside ``project_root``."""
    root_path = Path(project_root)
    definition_path = Path(definition)
    windows_definition_path = PureWindowsPath(definition)

    if definition.startswith("~"):
        raise ProjectDefinitionError(
            "[tool.crewai] definition must be a project-local path; "
            f"got {definition!r}."
        )

    if definition_path.is_absolute() or windows_definition_path.is_absolute():
        raise ProjectDefinitionError(
            "[tool.crewai] definition must be relative to the project root; "
            f"got {definition!r}."
        )

    try:
        root = root_path.resolve(strict=True)
    except OSError as exc:
        raise ProjectDefinitionError(
            f"Invalid project root for [tool.crewai] definition: {exc}"
        ) from exc

    candidate = root / definition_path
    try:
        resolved_candidate = candidate.resolve(strict=False)
    except OSError as exc:
        raise ProjectDefinitionError(
            f"Invalid [tool.crewai] definition path {definition!r}: {exc}"
        ) from exc

    if not resolved_candidate.is_relative_to(root):
        raise ProjectDefinitionError(
            "[tool.crewai] definition must resolve inside the project root; "
            f"got {definition!r}."
        )

    if not resolved_candidate.exists():
        raise ProjectDefinitionError(
            "[tool.crewai] definition must point to an existing file; "
            f"got {definition!r}."
        )

    if not resolved_candidate.is_file():
        raise ProjectDefinitionError(
            "[tool.crewai] definition must point to a regular file; "
            f"got {definition!r}."
        )

    return resolved_candidate


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
