import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.security.safe_path import (
    format_sandbox_error,
    validate_directory_path,
)


class FixedDirectoryReadToolSchema(BaseModel):
    """Input for DirectoryReadTool."""


class DirectoryReadToolSchema(FixedDirectoryReadToolSchema):
    """Input for DirectoryReadTool."""

    directory: str = Field(..., description="Mandatory directory to list content")


class DirectoryReadTool(BaseTool):
    """A tool for recursively listing a directory's files.

    Directories supplied at runtime must resolve inside ``base_dir`` (the
    current working directory by default), since they are typically chosen by
    an LLM. A ``directory`` given at construction time is developer-declared
    intent and is always allowed, even outside ``base_dir``, matching
    ``FileReadTool``. It is pinned at construction, so a later chdir cannot
    repoint it.

    Args:
        directory (Optional[str]): Directory to list. If provided, it becomes
            the only directory the tool lists.
        base_dir (Optional[str]): Directory that runtime paths must stay
            inside. Defaults to the current working directory.
        **kwargs: Additional keyword arguments passed to BaseTool.
    """

    name: str = "List files in directory"
    description: str = (
        "A tool that can be used to recursively list a directory's content."
    )
    args_schema: type[BaseModel] = DirectoryReadToolSchema
    directory: str | None = None
    base_dir: str | None = None

    # Pinned at construction so a later chdir cannot change which directory
    # the developer-declared default refers to.
    _declared_realpath: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        directory: str | None = None,
        base_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Anchor base_dir once, so the sandbox root cannot move under a later
        # chdir while the declared directory stays pinned.
        if base_dir is not None:
            base_dir = os.path.realpath(base_dir)
        self.base_dir = base_dir
        if directory is not None:
            self.directory = directory
            self._declared_realpath = os.path.realpath(
                os.path.join(base_dir or os.getcwd(), directory)
            )
            self.description = f"A tool that can be used to list {directory}'s content."
            self.args_schema = FixedDirectoryReadToolSchema
            self._generate_description()

    def _resolve_directory(self, directory: str) -> str:
        """Resolve *directory* and confirm the tool is allowed to list it.

        Raises:
            ValueError: If the path resolves outside ``base_dir`` or is not a
                directory.
        """
        declared = self._declared_realpath
        if declared is not None and directory in (self.directory, declared):
            if not os.path.isdir(declared):
                raise ValueError(f"Path '{self.directory}' is not a directory.")
            return declared
        try:
            return validate_directory_path(directory, self.base_dir)
        except ValueError as e:
            raise ValueError(
                format_sandbox_error(
                    e,
                    "Pass base_dir to DirectoryReadTool to allow listing another "
                    "directory tree.",
                )
            ) from e

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        directory: str | None = kwargs.get("directory", self.directory)
        if directory is None:
            raise ValueError("Directory must be provided.")

        directory = self._resolve_directory(directory)
        if directory[-1] == "/":
            directory = directory[:-1]
        files_list = [
            os.path.join(root, filename)
            for root, _, files in os.walk(directory)
            for filename in files
        ]
        files = "\n- ".join(files_list)
        return f"File paths: \n-{files}"
