"""The default store: the local filesystem, sandboxed to a base directory."""

from __future__ import annotations

from contextlib import AbstractContextManager
import os
from pathlib import Path
from typing import TextIO

from crewai_tools.security.safe_path import (
    format_error_for_display,
    format_path_for_display,
    validate_file_path,
)


class LocalFileStore:
    """Reads and writes the local filesystem.

    Containment is :func:`validate_file_path`: a resolved path must stay
    inside ``base_dir`` (the working directory by default), with symlinks and
    ``..`` segments resolved first.
    """

    label = "local filesystem"

    def resolve(self, path: str, base_dir: str | None = None) -> str:
        """Resolve *path*, confining it to *base_dir*."""
        return validate_file_path(path, base_dir)

    def normalize(self, path: str, base_dir: str | None = None) -> str:
        """Resolve *path* the way the sandbox does, without rejecting it.

        ``validate_file_path`` and ``format_path_for_display`` both join a
        relative path onto *base_dir* rather than the working directory.
        Normalization has to agree with them, or the same relative string
        would mean two different files.
        """
        if os.path.isabs(path):
            return os.path.realpath(path)
        base = os.path.realpath(base_dir) if base_dir is not None else os.getcwd()
        return os.path.realpath(os.path.join(base, path))

    def resolve_within(self, directory: str, filename: str) -> str:
        """Join *filename* under *directory*, blocking every escape route.

        ``..``, absolute paths and symlinks are all resolved before the
        check. ``is_relative_to`` compares whole path components, so it is
        safe on case-insensitive filesystems and avoids the "//" prefix edge
        case. A filename resolving to the directory itself (an empty
        filename, say) is not a valid file target.
        """
        root = Path(directory)
        try:
            resolved = Path(os.path.join(directory, filename)).resolve()
        except (OSError, ValueError) as exc:
            # e.g. an embedded null byte or an over-long name, which trip the
            # underlying syscall. str() on an OSError carries the absolute
            # filename, and the tools put this message straight into
            # agent-visible output, so strip it back to the reason.
            raise ValueError(format_error_for_display(exc)) from exc

        if not resolved.is_relative_to(root) or resolved == root:
            raise ValueError("the filename must not escape the target directory")
        return str(resolved)

    def display(self, resolved: str, base: str | None = None) -> str:
        """Return a path label with absolute prefixes stripped."""
        return format_path_for_display(resolved, base)

    def exists(self, resolved: str) -> bool:
        return os.path.exists(resolved)

    def ensure_parent(self, resolved: str) -> None:
        """Create the parent directory, including any missing ancestors."""
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)

    def open_text(self, resolved: str, encoding: str) -> AbstractContextManager[TextIO]:
        return open(resolved, "r", encoding=encoding)

    def write_text(
        self, resolved: str, content: str, encoding: str, *, overwrite: bool
    ) -> None:
        # "x" makes the create-exclusive check atomic, so an existence race
        # surfaces as FileExistsError rather than silently clobbering.
        mode = "w" if overwrite else "x"
        with open(resolved, mode, encoding=encoding) as handle:
            handle.write(content)
