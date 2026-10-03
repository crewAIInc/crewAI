from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.file_storage import FileStore, FileStoreError, resolve_file_store
from crewai_tools.security.safe_path import (
    format_error_for_display,
    format_sandbox_error,
)


def strtobool(val: str | bool) -> bool:
    """Coerce the spellings of true/false an LLM is likely to emit into a bool.

    Args:
        val: A bool, or one of y/yes/t/true/on/1 and n/no/f/false/off/0.

    Returns:
        The corresponding boolean.

    Raises:
        ValueError: If the string is not a recognized boolean spelling.
    """
    if isinstance(val, bool):
        return val
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid value to cast to bool: {val!r}")


class FileWriterToolInput(BaseModel):
    """Input for FileWriterTool."""

    filename: str = Field(
        ...,
        description=(
            "Name of the file to write, relative to 'directory'. May include "
            "subdirectories, which are created if they do not exist."
        ),
    )
    content: str = Field(..., description="The text content to write to the file.")
    directory: str | None = Field(
        "./",
        description=(
            "Directory to write the file into. A relative path resolves inside "
            "the tool's allowed directory, and defaults to its root. Created if "
            "it does not exist."
        ),
    )
    overwrite: str | bool = Field(
        False,
        description=(
            "Whether to replace the file when it already exists. Accepts "
            "true/false (also yes/no, on/off, 1/0). Defaults to false, which "
            "reports an error instead of replacing existing content."
        ),
    )


class FileWriterTool(BaseTool):
    """A tool for writing text content to a file.

    Writes are confined to ``base_dir`` (the current working directory by
    default), because the target directory and filename are typically chosen by
    an LLM at runtime. Set ``base_dir`` to widen that sandbox deliberately.

    Args:
        base_dir (Optional[str]): Directory that writes must stay inside.
            Defaults to the current working directory.
        encoding (str): Text encoding used to write the file. Defaults to UTF-8.

    Example:
        >>> tool = FileWriterTool()
        >>> tool.run(filename="report.md", content="# Report", overwrite=True)
        >>> # Allow the agent to write anywhere under /var/output:
        >>> tool = FileWriterTool(base_dir="/var/output")
    """

    name: str = "File Writer Tool"
    description: str = "A tool to write content to a specified file. Accepts filename, content, and optionally a directory path and overwrite flag as input. Writes are confined to the tool's allowed directory; a filename or directory that resolves outside it is rejected."
    args_schema: type[BaseModel] = FileWriterToolInput
    base_dir: str | None = None
    encoding: str = "utf-8"

    # Resolved once per tool: a deployment installs its store before the crew
    # is built, and swapping mid-run would change where a path points.
    _store: FileStore = PrivateAttr(default=None)  # type: ignore[assignment]

    def model_post_init(self, context: object) -> None:
        """Bind the store, then anchor base_dir with the store's own grammar.

        Anchoring cannot be a field validator: validators run before
        ``model_post_init``, so ``_store`` is not bound yet and the only option
        there is ``os.path.realpath`` — local-filesystem semantics applied to a
        path a remote store may not interpret that way at all. Doing it here
        keeps every path decision inside the seam, and still resolves once so a
        later chdir cannot move the sandbox.
        """
        super().model_post_init(context)
        self._store = resolve_file_store()
        if self.base_dir is not None:
            self.base_dir = self._store.normalize(self.base_dir)

    def _run(
        self,
        filename: str,
        content: str,
        directory: str | None = "./",
        overwrite: str | bool = False,
    ) -> str:
        """Write *content* to *filename*, confined to the tool's sandbox."""
        try:
            return self._write(filename, content, directory, overwrite)
        except (FileStoreError, OSError) as e:
            # A store can fail for reasons the local filesystem never had: an
            # unreachable endpoint, a size ceiling. Every other exit from this
            # tool is an agent-visible string, and raising here would kill the
            # agent's step rather than let it react, so this one is too. It
            # also covers the calls with no handler of their own — exists() and
            # the store's own path labelling — which is why it carries no path.
            return (
                f"An error occurred while writing to the file: the "
                f"{self._store.label} store failed. {format_error_for_display(e)}"
            )

    def _write(
        self,
        filename: str,
        content: str,
        directory: str | None,
        overwrite: str | bool,
    ) -> str:
        """Do the write. Wrapped by :meth:`_run`, which reports store failures."""
        directory = directory or "./"

        try:
            overwrite_file = strtobool(overwrite)
        except ValueError as e:
            return f"An error occurred while writing to the file: {e!s}"

        store = self._store

        # Confine the target directory to base_dir so an LLM-chosen directory
        # cannot reach outside the sandbox. The store also resolves symlinks
        # and ".." components before checking.
        try:
            resolved_directory = store.resolve(directory, self.base_dir)
        except ValueError as e:
            return "Error: Invalid directory: " + format_sandbox_error(
                e,
                "Pass base_dir to FileWriterTool to allow writing to another "
                "directory tree.",
            )

        # Then keep filename inside that directory.
        try:
            resolved_filepath = store.resolve_within(resolved_directory, filename)
        except ValueError as e:
            return f"Error: Invalid file path — {e!s}"

        display_filepath = store.display(resolved_filepath, resolved_directory)

        # Covers both a missing 'directory' and subdirectories inside 'filename'.
        try:
            store.ensure_parent(resolved_filepath)
        except FileExistsError:
            return (
                f"Error: Cannot write to {display_filepath} because a file already "
                f"exists where a directory is needed."
            )
        except OSError as e:
            return (
                f"Error: Could not create the directory for {display_filepath}. "
                f"{format_error_for_display(e)}"
            )

        if store.exists(resolved_filepath) and not overwrite_file:
            return f"File {display_filepath} already exists and overwrite option was not passed."

        try:
            store.write_text(
                resolved_filepath,
                content,
                self.encoding,
                overwrite=overwrite_file,
            )
        except FileExistsError:
            return f"File {display_filepath} already exists and overwrite option was not passed."
        except Exception as e:
            return (
                "An error occurred while writing to the file: "
                f"{format_error_for_display(e)}"
            )
        return f"Content successfully written to {display_filepath}"
