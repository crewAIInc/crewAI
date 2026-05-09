"""Helpers for detecting file returns from tools.

Tools can return file objects (instances of ``crewai_files.BaseFile`` or
subclasses such as ``File``, ``PDFFile``, ``ImageFile``, etc.) and the
agent executor will automatically attach those files to the conversation
so they become available as multimodal context for subsequent LLM calls.

This module exposes the pure helpers used by the executor to detect such
returns and convert them into a normalized representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai_files import BaseFile, FileInput


def _is_base_file(value: Any) -> bool:
    """Return True if ``value`` is a ``crewai_files.BaseFile`` subclass instance."""
    try:
        from crewai_files import BaseFile
    except ImportError:
        return False
    return isinstance(value, BaseFile)


def _file_key(file: BaseFile, used: set[str], default_index: int) -> str:
    """Pick a unique key for ``file`` when adding it to a files dict.

    Prefers the file's filename stem, falling back to ``file_<index>`` if no
    filename is available or if the chosen key collides with one already in
    use.
    """
    filename = file.filename
    base = Path(filename).stem if filename else f"file_{default_index}"
    candidate = base or f"file_{default_index}"
    if candidate not in used:
        return candidate

    counter = 1
    while True:
        candidate = f"{base}_{counter}" if base else f"file_{default_index}_{counter}"
        if candidate not in used:
            return candidate
        counter += 1


def extract_files_from_tool_result(
    result: Any,
) -> tuple[dict[str, FileInput] | None, str | None]:
    """Inspect a tool's return value and extract any ``FileInput`` instances.

    Tools may return:

    - A single ``BaseFile`` instance (``File``, ``PDFFile``, ``ImageFile``,
      ``TextFile``, ``AudioFile``, ``VideoFile``).
    - A list/tuple of ``BaseFile`` instances.
    - A dict mapping names to ``BaseFile`` instances.

    When any of these shapes are detected this returns a tuple
    ``(files, message)`` where ``files`` is a dict suitable for the
    multimodal ``files`` slot on a user message and ``message`` is a short
    confirmation string describing what was added (intended to be shown to
    the LLM as the textual tool result).

    For any other return type the helper returns ``(None, None)`` so the
    caller can keep the existing string-based behavior unchanged.

    Args:
        result: The raw return value of a tool's ``run`` / ``_run`` method.

    Returns:
        A ``(files, message)`` tuple. ``files`` is ``None`` when no files
        were detected.
    """
    files: dict[str, BaseFile] = {}

    if _is_base_file(result):
        key = _file_key(result, used=set(), default_index=0)
        files[key] = result
    elif isinstance(result, Mapping) and result:
        if not all(_is_base_file(value) for value in result.values()):
            return None, None
        used: set[str] = set()
        for raw_key, value in result.items():
            key = str(raw_key)
            if not key or key in used:
                key = _file_key(value, used=used, default_index=len(used))
            used.add(key)
            files[key] = value
    elif (
        isinstance(result, Sequence)
        and not isinstance(result, (str, bytes, bytearray))
        and result
    ):
        if not all(_is_base_file(value) for value in result):
            return None, None
        used = set()
        for index, value in enumerate(result):
            key = _file_key(value, used=used, default_index=index)
            used.add(key)
            files[key] = value
    else:
        return None, None

    if not files:
        return None, None

    message = _format_files_message(files)
    return files, message


def _format_files_message(files: Mapping[str, BaseFile]) -> str:
    """Build a confirmation string describing the files that were added."""
    descriptions: list[str] = []
    for key, file in files.items():
        filename = file.filename
        try:
            content_type = file.content_type
        except Exception:  # pragma: no cover - defensive
            content_type = "unknown"
        if filename:
            descriptions.append(f"'{key}' ({filename}, {content_type})")
        else:
            descriptions.append(f"'{key}' ({content_type})")

    count = len(files)
    suffix = "" if count == 1 else "s"
    return (
        f"Added {count} file{suffix} to the agent context: "
        + ", ".join(descriptions)
        + ". They are available for subsequent reasoning and tool calls."
    )
