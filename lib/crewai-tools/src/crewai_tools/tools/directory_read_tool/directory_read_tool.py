import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.security.safe_path import (
    format_error_for_display,
    format_sandbox_error,
    validate_directory_path,
)


class FixedDirectoryReadToolSchema(BaseModel):
    """Input for DirectoryReadTool."""


class DirectoryReadToolSchema(FixedDirectoryReadToolSchema):
    """Input for DirectoryReadTool."""

    directory: str = Field(..., description="Mandatory directory to list content")


class DirectoryReadTool(BaseTool):
    """A tool for recursively listing a directory's content.

    A directory given at construction time is intent declared by the
    developer, so it is always allowed past the containment check, even when
    it lives outside base_dir (the tool's operation can still fail if the
    path does not exist). It is pinned at construction, so a later chdir
    cannot repoint it. A directory supplied at runtime, typically chosen by
    an LLM, must resolve inside base_dir (the current working directory by
    default).

    Args:
        directory (Optional[str]): Directory to list. If provided, this
            becomes the fixed directory for the tool and the LLM can no
            longer choose a different one.
        base_dir (Optional[str]): Directory that runtime supplied paths
            must stay inside. Defaults to the current working directory.
        **kwargs: Additional keyword arguments passed to BaseTool.
    """

    name: str = "List files in directory"
    description: str = (
        "A tool that can be used to recursively list a directory's content."
    )
    args_schema: type[BaseModel] = DirectoryReadToolSchema
    directory: str | None = None
    base_dir: str | None = None

    _declared_realpath: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        directory: str | None = None,
        base_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if base_dir is not None:
            base_dir = os.path.realpath(base_dir)
        self.base_dir = base_dir

        if directory is not None:
            self.directory = directory
            self.description = f"A tool that can be used to list {directory}'s content."
            self.args_schema = FixedDirectoryReadToolSchema
            self._generate_description()
            # Resolve now so a later chdir can't move the declared directory
            # and so it always passes the containment check below.
            self._declared_realpath = (
                os.path.realpath(directory)
                if os.path.isabs(directory)
                else os.path.realpath(os.path.join(base_dir or os.getcwd(), directory))
            )

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        directory: str | None = kwargs.get("directory", self.directory)
        if directory is None:
            raise ValueError("Directory must be provided.")

        if self._declared_realpath is not None and directory == self.directory:
            directory = self._declared_realpath
            if not os.path.isdir(directory):
                return f"Error: '{directory}' is not a directory."
        else:
            try:
                directory = validate_directory_path(directory, self.base_dir)
            except ValueError as e:
                return "Error: Invalid directory: " + format_sandbox_error(
                    e,
                    "Pass base_dir to DirectoryReadTool to allow listing another "
                    "directory tree.",
                )

        directory = directory.rstrip("/") or "/"
        walk_errors: list[OSError] = []
        files_list = [
            os.path.join(directory, os.path.relpath(os.path.join(root, filename), directory))
            for root, dirs, files in os.walk(directory, onerror=walk_errors.append)
            for filename in files
        ]
        if walk_errors and not files_list:
            # os.walk() does not raise on its own. It only calls onerror if
            # given one, then moves on. Without this check a missing or
            # unreadable directory would silently return an empty listing.
            return f"Error: Could not list '{directory}'. {format_error_for_display(walk_errors[0])}"
        files = "\n- ".join(files_list)
        return f"File paths: \n-{files}"
