"""TOML / pyproject.toml utilities shared by crewai and crewai-cli."""

from __future__ import annotations

from functools import reduce
from pathlib import Path, PureWindowsPath
import sys
from typing import Any
import uuid

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

    if "definition" not in crewai_config:
        return None
    raw_definition = crewai_config["definition"]
    if not isinstance(raw_definition, str):
        raise ProjectDefinitionError(
            "[tool.crewai] definition must be a string project-local path; "
            f"got {raw_definition!r}."
        )

    definition = raw_definition.strip()
    if not definition:
        raise ProjectDefinitionError(
            "[tool.crewai] definition must be a non-empty project-local path."
        )

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


_PROJECT_ID_KEY = "project_id"


def get_project_id(pyproject_path: str | Path = "pyproject.toml") -> str | None:
    """Return ``[tool.crewai].project_id`` if the project has one.

    Read-only and safe to call from library code: it never creates or modifies
    anything. Use this everywhere except the CLI commands that are allowed to
    mint an id (see :func:`get_or_create_project_id`).

    Args:
        pyproject_path: Path to the project's ``pyproject.toml``.

    Returns:
        The project id, or None when the file is missing, unreadable, or has
        no id configured.
    """
    try:
        pyproject_data = read_toml(pyproject_path)
    except (OSError, tomli.TOMLDecodeError):
        return None

    project_id = get_crewai_project_config(pyproject_data).get(_PROJECT_ID_KEY)
    return project_id if isinstance(project_id, str) and project_id else None


def get_or_create_project_id(
    pyproject_path: str | Path = "pyproject.toml",
) -> tuple[str | None, bool]:
    """Return the project's id, minting and persisting one if absent.

    Writes ``project_id`` into the ``[tool.crewai]`` table so it is committed
    with the repository. That makes it stable across machines, teammates, CI,
    and containers - unlike a machine- or user-derived identifier.

    Only CLI commands the user explicitly invoked should call this. Library
    code must use :func:`get_project_id` instead; silently rewriting a user's
    ``pyproject.toml`` during ``Crew.kickoff()`` would be surprising.

    Args:
        pyproject_path: Path to the project's ``pyproject.toml``.

    Returns:
        A ``(project_id, created)`` tuple. ``created`` is True only when an id
        was minted and written on this call, so callers can tell the user. Both
        values are ``(None, False)`` when the file is missing or not writable -
        this is best-effort and never raises.
    """
    existing = get_project_id(pyproject_path)
    if existing:
        return existing, False

    path = Path(pyproject_path)
    if not path.is_file():
        return None, False

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None, False

    project_id = str(uuid.uuid4())
    updated = _insert_project_id(content, project_id)
    if updated is None:
        return None, False

    try:
        path.write_text(updated, encoding="utf-8")
    except OSError:
        # Read-only checkout, permissions, container FS - not worth failing over.
        return None, False

    return project_id, True


def _insert_project_id(content: str, project_id: str) -> str | None:
    """Add ``project_id`` to the ``[tool.crewai]`` table in TOML source text.

    Edits the raw text rather than round-tripping through a TOML writer so
    formatting, ordering, and comments in the rest of the file are preserved.

    Args:
        content: Full contents of a ``pyproject.toml``.
        project_id: The id to insert.

    Returns:
        Updated file contents, or None if the edit could not be made safely.
    """
    lines = content.splitlines(keepends=True)
    entry = f'{_PROJECT_ID_KEY} = "{project_id}"\n'

    for index, line in enumerate(lines):
        if line.strip() != "[tool.crewai]":
            continue

        # Insert at the end of the table, before the next table header, so the
        # key cannot land inside a different section.
        insert_at = len(lines)
        for offset in range(index + 1, len(lines)):
            if lines[offset].lstrip().startswith("["):
                insert_at = offset
                break

        # Step back over trailing blank lines so the key stays in the table.
        while insert_at > index + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1

        if insert_at > 0 and not lines[insert_at - 1].endswith("\n"):
            lines[insert_at - 1] += "\n"

        lines.insert(insert_at, entry)
        return "".join(lines)

    # No [tool.crewai] table: append one rather than guessing where it belongs.
    suffix = "" if content.endswith("\n") or not content else "\n"
    return f"{content}{suffix}\n[tool.crewai]\n{entry}"
