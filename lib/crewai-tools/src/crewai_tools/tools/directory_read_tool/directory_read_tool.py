import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FixedDirectoryReadToolSchema(BaseModel):
    """Input for DirectoryReadTool."""


class DirectoryReadToolSchema(FixedDirectoryReadToolSchema):
    """Input for DirectoryReadTool."""

    directory: str = Field(..., description="Mandatory directory to list content")


class DirectoryReadTool(BaseTool):
    name: str = "List files in directory"
    description: str = (
        "A tool that can be used to recursively list a directory's content."
    )
    args_schema: type[BaseModel] = DirectoryReadToolSchema
    directory: str | None = None

    def __init__(self, directory: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if directory is not None:
            self.directory = directory
            self.description = f"A tool that can be used to list {directory}'s content."
            self.args_schema = FixedDirectoryReadToolSchema
            self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        directory: str | None = kwargs.get("directory", self.directory)
        if directory is None:
            raise ValueError("Directory must be provided.")

        if directory[-1] == "/":
            directory = directory[:-1]
        files_list = [
            f"{directory}/{(os.path.join(root, filename).replace(directory, '').lstrip(os.path.sep))}"
            for root, dirs, files in os.walk(directory)
            for filename in files
        ]
        files = "\n- ".join(files_list)
        return f"File paths: \n-{files}"
