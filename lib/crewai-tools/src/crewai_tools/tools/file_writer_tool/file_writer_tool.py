import os
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai_tools.security.safe_path import (
    format_error_for_display,
    format_path_for_display,
    validate_file_path,
)


def strtobool(val: str | bool) -> bool:
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
            "Directory to write the file into. Created if it does not exist. "
            "Defaults to the current working directory."
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
    description: str = "A tool to write content to a specified file. Accepts filename, content, and optionally a directory path and overwrite flag as input."
    args_schema: type[BaseModel] = FileWriterToolInput
    base_dir: str | None = None
    encoding: str = "utf-8"

    def _run(
        self,
        filename: str,
        content: str,
        directory: str | None = "./",
        overwrite: str | bool = False,
    ) -> str:
        directory = directory or "./"

        try:
            overwrite_file = strtobool(overwrite)
        except ValueError as e:
            return f"An error occurred while writing to the file: {e!s}"

        # Confine the target directory to base_dir so an LLM-chosen directory
        # cannot reach outside the sandbox. validate_file_path also resolves
        # symlinks and ".." components.
        try:
            resolved_directory = Path(validate_file_path(directory, self.base_dir))
        except ValueError as e:
            return f"Error: Invalid directory: {e!s}"

        # Keep filename inside the target directory, blocking "..", absolute
        # paths and symlink escapes. is_relative_to() compares whole path
        # components, so it is safe on case-insensitive filesystems and avoids
        # the "//" prefix edge case. A filepath that resolves to the directory
        # itself (e.g. an empty filename) is not a valid file target.
        resolved_filepath = Path(os.path.join(resolved_directory, filename)).resolve()
        display_filepath = format_path_for_display(
            str(resolved_filepath), str(resolved_directory)
        )
        if (
            not resolved_filepath.is_relative_to(resolved_directory)
            or resolved_filepath == resolved_directory
        ):
            return "Error: Invalid file path — the filename must not escape the target directory."

        # Covers both a missing 'directory' and subdirectories inside 'filename'.
        try:
            os.makedirs(resolved_filepath.parent, exist_ok=True)
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

        if resolved_filepath.exists() and not overwrite_file:
            return f"File {display_filepath} already exists and overwrite option was not passed."

        mode = "w" if overwrite_file else "x"
        try:
            with open(resolved_filepath, mode, encoding=self.encoding) as file:
                file.write(content)
        except FileExistsError:
            return f"File {display_filepath} already exists and overwrite option was not passed."
        except Exception as e:
            return (
                "An error occurred while writing to the file: "
                f"{format_error_for_display(e)}"
            )
        return f"Content successfully written to {display_filepath}"
