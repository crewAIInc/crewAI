"""Tool for reading input files provided to the crew."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from crewai.tools.base_tool import BaseTool


if TYPE_CHECKING:
    from crewai_files import FileInput


class ReadFileToolSchema(BaseModel):
    """Schema for read file tool arguments."""

    file_name: str = Field(..., description="The name of the input file to read")


class ReadFileTool(BaseTool):
    """Tool for reading input files provided to the crew kickoff.

    Provides agents access to files passed via the `files` key in inputs.
    """

    name: str = "read_file"
    description: str = (
        "Read content from an input file by name. "
        "Returns file content as text for text files, or base64 for binary files."
    )
    args_schema: type[BaseModel] = ReadFileToolSchema

    _files: dict[str, FileInput] | None = PrivateAttr(default=None)

    def set_files(self, files: dict[str, FileInput] | None) -> None:
        """Set available input files.

        Args:
            files: Dictionary mapping file names to file inputs.
        """
        self._files = files

    def _run(self, file_name: str, **kwargs: object) -> str:
        """Read an input file by name.

        Args:
            file_name: The name of the file to read.

        Returns:
            File content as text for text files, or base64 encoded for binary.
        """
        if not self._files:
            return "No input files available."

        if file_name not in self._files:
            available = ", ".join(self._files.keys())
            return f"File '{file_name}' not found. Available files: {available}"

        file_input = self._files[file_name]
        content = file_input.read()
        content_type = file_input.content_type
        filename = file_input.filename or file_name

        text_types = (
            "text/",
            "application/json",
            "application/xml",
            "application/x-yaml",
        )

        if any(content_type.startswith(t) for t in text_types):
            return content.decode("utf-8")

        encoded = base64.b64encode(content).decode("ascii")
        return f"[Binary file: {filename} ({content_type})]\nBase64: {encoded}"
