import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel

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
    filename: str
    directory: str | None = "./"
    overwrite: str | bool = False
    content: str


class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = "A tool to write content to a specified file. Accepts filename, content, and optionally a directory path and overwrite flag as input."
    args_schema: type[BaseModel] = FileWriterToolInput

    def _run(self, **kwargs: Any) -> str:
        try:
            directory = kwargs.get("directory") or "./"
            filename = kwargs["filename"]

            filepath = os.path.join(directory, filename)

            # Confine the resolved write target to an allow-listed root
            # (cwd + CREWAI_TOOLS_ALLOWED_DIRS), NOT merely inside the
            # caller-supplied `directory`. That value is itself untrusted when
            # an LLM tool call chooses it, so checking containment against it
            # would let an agent write anywhere (e.g. ~/.ssh/authorized_keys).
            # validate_file_path resolves symlinks and ".." before checking.
            try:
                real_filepath = validate_file_path(filepath)
            except ValueError as e:
                return f"Error: {format_error_for_display(e)}"

            real_directory = os.path.dirname(real_filepath)
            display_filepath = format_path_for_display(real_filepath, real_directory)

            # A target that resolves to an existing directory is not a valid
            # file destination.
            if os.path.isdir(real_filepath):
                return (
                    "Error: Invalid file path — the target must be a file, "
                    "not a directory."
                )

            if kwargs.get("directory"):
                os.makedirs(real_directory, exist_ok=True)

            kwargs["overwrite"] = strtobool(kwargs["overwrite"])

            if os.path.exists(real_filepath) and not kwargs["overwrite"]:
                return f"File {display_filepath} already exists and overwrite option was not passed."

            mode = "w" if kwargs["overwrite"] else "x"
            with open(real_filepath, mode) as file:
                file.write(kwargs["content"])
            return f"Content successfully written to {display_filepath}"
        except FileExistsError:
            return f"File {display_filepath} already exists and overwrite option was not passed."
        except KeyError as e:
            return f"An error occurred while accessing key: {e!s}"
        except Exception as e:
            return (
                "An error occurred while writing to the file: "
                f"{format_error_for_display(e)}"
            )
