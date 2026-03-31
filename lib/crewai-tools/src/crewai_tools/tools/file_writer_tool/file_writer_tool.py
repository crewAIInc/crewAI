import os
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel


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

            # Prevent path traversal: the resolved path must be strictly inside
            # the resolved directory. This blocks ../sequences, absolute paths in
            # filename, and symlink escapes regardless of how directory is set.
            # is_relative_to() does a proper path-component comparison that is
            # safe on case-insensitive filesystems and avoids the "// " edge case
            # that plagues startswith(real_directory + os.sep).
            # We also reject the case where filepath resolves to the directory
            # itself, since that is not a valid file target.
            real_directory = Path(directory).resolve()
            real_filepath = Path(filepath).resolve()
            if (
                not real_filepath.is_relative_to(real_directory)
                or real_filepath == real_directory
            ):
                return "Error: Invalid file path — the filename must not escape the target directory."

            if kwargs.get("directory"):
                os.makedirs(real_directory, exist_ok=True)

            kwargs["overwrite"] = strtobool(kwargs["overwrite"])

            if os.path.exists(real_filepath) and not kwargs["overwrite"]:
                return f"File {real_filepath} already exists and overwrite option was not passed."

            mode = "w" if kwargs["overwrite"] else "x"
            with open(real_filepath, mode) as file:
                file.write(kwargs["content"])
            return f"Content successfully written to {real_filepath}"
        except FileExistsError:
            return f"File {real_filepath} already exists and overwrite option was not passed."
        except KeyError as e:
            return f"An error occurred while accessing key: {e!s}"
        except Exception as e:
            return f"An error occurred while writing to the file: {e!s}"
