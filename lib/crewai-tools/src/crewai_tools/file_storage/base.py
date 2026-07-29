"""The store protocol the file tools are written against."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, TextIO, runtime_checkable


class FileStoreError(Exception):
    """A store failed for a reason with no stdlib exception that fits.

    Stores should prefer the built-in filesystem exceptions where one
    applies — ``FileNotFoundError``, ``PermissionError``,
    ``IsADirectoryError``, ``FileExistsError`` — because the tools already
    translate those into their established messages. Raise this only for
    failures specific to the backing service, such as an unreachable
    endpoint or a size limit the local filesystem does not have.
    """


@runtime_checkable
class FileStore(Protocol):
    """Where :class:`FileReadTool` and :class:`FileWriterTool` do their I/O.

    Paths crossing this boundary are *store paths*: whatever ``resolve``
    returned. For the local store those are absolute filesystem paths; for a
    remote store they may be keys or workspace-relative paths. The tools
    never interpret them, they only pass them back.
    """

    #: Short human-readable name, used in error messages so a failure makes
    #: clear which store produced it (e.g. ``"local filesystem"``).
    label: str

    def resolve(self, path: str, base_dir: str | None = None) -> str:
        """Normalize *path* and confirm the caller may touch it.

        Args:
            path: The caller-supplied path, absolute or relative.
            base_dir: Optional containment root supplied by the tool.

        Returns:
            The store path to use for subsequent calls.

        Raises:
            ValueError: If the path falls outside what the store allows.
        """

    def normalize(self, path: str, base_dir: str | None = None) -> str:
        """Normalize *path* for identity comparison, without containment.

        :class:`FileReadTool` uses this to pin the file declared at
        construction, so it can recognize that path again later even if the
        working directory has since moved. Unlike :meth:`resolve` it never
        rejects: a path outside the sandbox still has a canonical form.
        """

    def resolve_within(self, directory: str, filename: str) -> str:
        """Join *filename* under the already-resolved *directory*.

        Kept separate from :meth:`resolve` because the writer applies two
        levels of containment: the directory must be inside the store's
        sandbox, and the filename must then stay inside that directory.

        Raises:
            ValueError: If *filename* escapes *directory*, or names the
                directory itself.
        """

    def display(self, resolved: str, base: str | None = None) -> str:
        """Return a label for *resolved* that is safe to show an LLM.

        Must not leak absolute directory prefixes; the tools put the result
        straight into agent-visible output.
        """

    def exists(self, resolved: str) -> bool:
        """Whether something already lives at *resolved*."""

    def ensure_parent(self, resolved: str) -> None:
        """Create the container *resolved* will live in, if it needs one.

        Raises:
            FileExistsError: If a non-container already occupies that name.
        """

    def open_text(self, resolved: str, encoding: str) -> AbstractContextManager[TextIO]:
        """Open *resolved* for reading as text.

        Returning a file-like object rather than a string keeps the local
        store lazy, so reading a small window out of a huge file does not
        pull the whole thing into memory. Remote stores that must fetch
        eagerly can wrap the payload in ``io.StringIO``.
        """

    def write_text(
        self, resolved: str, content: str, encoding: str, *, overwrite: bool
    ) -> None:
        """Write *content* to *resolved*.

        Raises:
            FileExistsError: If the path exists and *overwrite* is false.
        """
