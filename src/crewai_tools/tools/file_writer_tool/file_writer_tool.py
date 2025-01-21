import os
from distutils.util import strtobool
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel


class FileWriterToolInput(BaseModel):
    filename: str
    directory: Optional[str] = "./"
    overwrite: str = "False"
    content: str


class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = "A tool to write content to a specified file. Accepts filename, content, and optionally a directory path and overwrite flag as input."
    args_schema: Type[BaseModel] = FileWriterToolInput

    def _run(self, **kwargs: Any) -> str:
        try:
            # Create the directory if it doesn't exist
            if kwargs.get("directory") and not os.path.exists(kwargs["directory"]):
                os.makedirs(kwargs["directory"])

            # Construct the full path
            filepath = os.path.join(kwargs.get("directory") or "", kwargs["filename"])

            # Convert overwrite to boolean
            kwargs["overwrite"] = bool(strtobool(kwargs["overwrite"]))

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
            return f"An error occurred while accessing key: {str(e)}"
        except Exception as e:
            return f"An error occurred while writing to the file: {str(e)}"
