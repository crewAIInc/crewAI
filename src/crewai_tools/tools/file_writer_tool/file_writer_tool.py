import os
from typing import Any, Optional, Type

from pydantic import BaseModel

from ..base_tool import BaseTool


class FileWriterToolInput(BaseModel):
    filename: str
    content: str
    directory: Optional[str] = None
    overwrite: bool = False


class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = (
        "A tool to write content to a specified file. Accepts filename, content, and optionally a directory path and overwrite flag as input."
    )
    args_schema: Type[BaseModel] = FileWriterToolInput

    def _run(self, **kwargs: Any) -> str:
        try:
            # Create the directory if it doesn't exist
            if kwargs["directory"] and not os.path.exists(kwargs["directory"]):
                os.makedirs(kwargs["directory"])

            # Construct the full path
            filepath = os.path.join(kwargs["directory"] or "", kwargs["filename"])

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
        except Exception as e:
            return f"An error occurred while writing to the file: {str(e)}"
