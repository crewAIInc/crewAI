"""TOML / pyproject.toml utilities shared by crewai and crewai-cli."""

from __future__ import annotations

from functools import reduce
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile
from typing import Any
import uuid

from rich.console import Console
import tomli

from crewai_core.lock_store import lock as store_lock


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

    return _usable_project_id(get_crewai_project_config(pyproject_data))


def _has_crewai_table(pyproject_data: dict[str, Any]) -> bool:
    """True if ``[tool.crewai]`` exists, even when empty.

    Distinguishes "declared but empty" from "absent", which
    :func:`get_crewai_project_config` cannot: it returns ``{}`` for both.
    """
    tool_config = pyproject_data.get("tool")
    return isinstance(tool_config, dict) and isinstance(tool_config.get("crewai"), dict)


def _usable_project_id(crewai_config: dict[str, Any]) -> str | None:
    """Return the configured id if it is usable as an identifier.

    Whitespace-only values are treated as absent: they are truthy in Python but
    are not an identity, and would otherwise propagate into login payloads and
    tracing context.
    """
    project_id = crewai_config.get(_PROJECT_ID_KEY)
    if not isinstance(project_id, str):
        return None
    stripped = project_id.strip()
    return stripped or None


def get_or_create_project_id(
    pyproject_path: str | Path = "pyproject.toml",
) -> str | None:
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
        The project id, or None when ``pyproject.toml`` is missing, malformed,
        or not writable. Best-effort - never raises.
    """
    path = Path(pyproject_path)
    if not path.is_file():
        return None

    # Cross-process lock: two CLI invocations could otherwise both see no id,
    # mint different uuids, and clobber each other - leaving one caller holding
    # an id that is not the one on disk.
    try:
        with store_lock(_project_id_lock_name(path)):
            return _get_or_create_project_id_locked(path)
    except Exception:
        # Lock backend unavailable; a torn write is worse than no id.
        return get_project_id(path)


def _project_id_lock_name(path: Path) -> str:
    """Return a stable lock name for a project's ``pyproject.toml``."""
    return f"file:{os.path.realpath(path)}"


def _get_or_create_project_id_locked(path: Path) -> str | None:
    """Read-modify-write the project id while holding the lock.

    Re-reads under the lock so a concurrent minter's id is returned rather than
    overwritten.
    """
    try:
        content = _read_preserving_newlines(path)
    except OSError:
        return None

    # Parse here rather than relying on get_project_id, which reports malformed
    # files and absent ids identically. Appending to a file we cannot parse
    # would corrupt it further, so bail instead.
    try:
        pyproject_data = parse_toml(content)
    except (tomli.TOMLDecodeError, ValueError):
        return None

    crewai_config = get_crewai_project_config(pyproject_data)
    existing = _usable_project_id(crewai_config)
    if existing:
        return existing

    # Only ever add a key to an existing [tool.crewai] table. Creating the table
    # would rewrite the pyproject.toml of any directory that merely happens to
    # have one, which `crewai run` could otherwise do before it has established
    # that the cwd is a CrewAI project at all.
    #
    # Presence, not truthiness: an empty `[tool.crewai]` table is still a CrewAI
    # marker, and get_crewai_project_config returns {} for both cases.
    if not _has_crewai_table(pyproject_data):
        return None

    project_id = str(uuid.uuid4())
    updated = _set_project_id(content, project_id)
    if updated is None:
        return None

    # Verify before writing: never leave a project with unparsable TOML because
    # of this feature.
    try:
        parse_toml(updated)
    except (tomli.TOMLDecodeError, ValueError):
        return None

    # Checked explicitly: os.replace only needs a writable *directory*, so an
    # atomic write would happily overwrite a file the user marked read-only.
    if not os.access(path, os.W_OK):
        return None

    try:
        _write_atomically(path, updated)
    except OSError:
        # Read-only checkout, permissions, container FS - not worth failing over.
        return None

    return project_id


def _read_preserving_newlines(path: Path) -> str:
    """Read text without translating line endings.

    ``Path.read_text`` normalizes CRLF to LF, so a later write would silently
    convert a CRLF-committed file to LF and show up as a whole-file diff.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_atomically(path: Path, content: str) -> None:
    """Replace ``path`` with ``content`` via a temp file in the same directory.

    An interrupted or concurrent write must never leave a truncated
    ``pyproject.toml`` behind.
    """
    directory = path.parent
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=directory,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _is_table_header(line: str, table: str) -> bool:
    """True if ``line`` opens ``table``, tolerating a trailing inline comment.

    ``[tool.crewai]  # config`` is valid TOML. Comparing the stripped line to
    the header verbatim would miss it, and the caller would then append a second
    ``[tool.crewai]`` header - a duplicate table definition, which is invalid
    TOML.
    """
    stripped = line.strip()
    if not stripped.startswith("["):
        return False
    closing = stripped.find("]")
    if closing == -1:
        return False
    if stripped[: closing + 1] != table:
        return False
    remainder = stripped[closing + 1 :].strip()
    return remainder == "" or remainder.startswith("#")


def _is_any_table_header(line: str) -> bool:
    """True if ``line`` opens any TOML table or array-of-tables."""
    return line.lstrip().startswith("[")


def _project_id_key_index(lines: list[str], start: int, end: int) -> int | None:
    """Return the index of an existing ``project_id`` assignment in a range."""
    for index in range(start, end):
        candidate = lines[index].strip()
        if not candidate or candidate.startswith("#"):
            continue
        key, separator, _ = candidate.partition("=")
        if separator and key.strip().strip("\"'") == _PROJECT_ID_KEY:
            return index
    return None


def _set_project_id(content: str, project_id: str) -> str | None:
    """Set ``project_id`` in the ``[tool.crewai]`` table of TOML source text.

    Replaces an existing ``project_id`` assignment rather than adding a second
    one: a blank or non-string value reads as "absent", and appending in that
    case would produce a duplicate key and therefore invalid TOML.

    Edits the raw text rather than round-tripping through a TOML writer so
    formatting, ordering, and comments in the rest of the file are preserved.

    Args:
        content: Full contents of a ``pyproject.toml``.
        project_id: The id to set.

    Returns:
        Updated file contents, or None if the edit could not be made safely.
    """
    lines = content.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in content else "\n"
    entry = f'{_PROJECT_ID_KEY} = "{project_id}"{newline}'

    for index, line in enumerate(lines):
        if not _is_table_header(line, "[tool.crewai]"):
            continue

        # Bound the table: everything up to the next table header.
        table_end = len(lines)
        for offset in range(index + 1, len(lines)):
            if _is_any_table_header(lines[offset]):
                table_end = offset
                break

        existing = _project_id_key_index(lines, index + 1, table_end)
        if existing is not None:
            lines[existing] = entry
            return "".join(lines)

        # Step back over trailing blank lines so the key stays in the table.
        insert_at = table_end
        while insert_at > index + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1

        if insert_at > 0 and not lines[insert_at - 1].endswith(("\n", "\r")):
            lines[insert_at - 1] += newline

        lines.insert(insert_at, entry)
        return "".join(lines)

    # No [tool.crewai] table. Never create one: that would let this feature
    # rewrite the pyproject.toml of a directory that is not a CrewAI project.
    return None
