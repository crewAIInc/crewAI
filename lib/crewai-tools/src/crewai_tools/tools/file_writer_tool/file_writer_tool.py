import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel


def strtobool(val) -> bool:
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
            # Create the directory if it doesn't exist
            if kwargs.get("directory") and not os.path.exists(kwargs["directory"]):
                os.makedirs(kwargs["directory"])

            # Construct the full path
            filepath = os.path.join(kwargs.get("directory") or "", kwargs["filename"])

            # Convert overwrite to boolean
            kwargs["overwrite"] = strtobool(kwargs["overwrite"])

            # Check if file exists and overwrite is not allowed
            if os.path.exists(filepath) and not kwargs["overwrite"]:
                return f"File {filepath} already exists and overwrite option was not passed."

            # Write content to the file
            mode = "w" if kwargs["overwrite"] else "x"
            with open(filepath, mode) as file:
                file.write(kwargs["content"])
            return f"Content successfully written to {filepath}"
        except FileExistsError:
            return (
                f"File {filepath} already exists and overwrite option was not passed."
            )
        except KeyError as e:
            return f"An error occurred while accessing key: {e!s}"
        except Exception as e:
            return f"An error occurred while writing to the file: {e!s}"
