from itertools import islice
import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.security.safe_path import (
    format_error_for_display,
    format_path_for_display,
    format_sandbox_error,
    validate_file_path,
)


def _resolve_against_base(path: str, base_dir: str | None) -> str:
    """Resolve *path* the way the sandbox does, anchoring relatives to *base_dir*.

    ``validate_file_path`` and ``format_path_for_display`` both join a relative
    path onto *base_dir* rather than the working directory. Resolution has to
    agree with them, or the same relative string would mean two different files.

    Args:
        path: The path to resolve.
        base_dir: The anchor for relative paths. Defaults to the working directory.

    Returns:
        The resolved absolute path.
    """
    if os.path.isabs(path):
        return os.path.realpath(path)
    base = os.path.realpath(base_dir) if base_dir is not None else os.getcwd()
    return os.path.realpath(os.path.join(base, path))


class FileReadToolSchema(BaseModel):
    """Input for FileReadTool."""

    file_path: str | None = Field(
        None,
        description=(
            "Full path of the file to read. Omit it to read the tool's default "
            "file, which only works when one was configured."
        ),
    )
    start_line: int | None = Field(
        1, description="Line number to start reading from (1-indexed)"
    )
    line_count: int | None = Field(
        None, description="Number of lines to read. If None, reads the entire file"
    )


class FileReadTool(BaseTool):
    """A tool for reading file contents.

    This tool inherits its schema handling from BaseTool to avoid recursive schema
    definition issues. The args_schema is set to FileReadToolSchema, whose
    file_path parameter is optional so the tool's default file can be read by
    omitting it. The schema should not be overridden in the constructor as it
    would break the inheritance chain and cause infinite loops.

    The tool supports two ways of specifying the file path:
    1. At construction time via the file_path parameter
    2. At runtime via the file_path parameter in the tool's input

    Paths supplied at runtime must resolve inside ``base_dir`` (the current
    working directory by default), since they are typically chosen by an LLM.
    A ``file_path`` given at construction time is developer-declared intent and
    is always allowed past the containment check, even when it lives outside
    ``base_dir`` (the read itself can still fail). It is pinned at
    construction, so a later chdir cannot repoint it, and it can be addressed
    either by omitting ``file_path`` or by the label shown in the description.

    Args:
        file_path (Optional[str]): Path to the file to be read. If provided,
            this becomes the default file path for the tool.
        base_dir (Optional[str]): Directory that runtime paths must stay inside.
            Defaults to the current working directory.
        encoding (str): Text encoding used to decode the file. Defaults to UTF-8.
        **kwargs: Additional keyword arguments passed to BaseTool.

    Example:
        >>> tool = FileReadTool(file_path="/path/to/file.txt")
        >>> content = tool.run()  # Reads /path/to/file.txt
        >>> content = tool.run(file_path="/path/to/other.txt")  # Reads other.txt
        >>> content = tool.run(
        ...     file_path="/path/to/file.txt", start_line=100, line_count=50
        ... )  # Reads lines 100-149
        >>> # Widen the sandbox so the agent may read anything under /data:
        >>> tool = FileReadTool(base_dir="/data")
    """

    name: str = "Read a file's content"
    description: str = "A tool that reads the content of a file. To use this tool, provide a 'file_path' parameter with the path to the file you want to read. Reads are confined to the tool's allowed directory; a path that resolves outside it is rejected. Optionally, provide 'start_line' to start reading from a specific line and 'line_count' to limit the number of lines read."
    args_schema: type[BaseModel] = FileReadToolSchema
    file_path: str | None = None
    base_dir: str | None = None
    encoding: str = "utf-8"

    # Pinned at construction so a later chdir cannot change which file the
    # developer-declared default refers to.
    _declared_realpath: str | None = PrivateAttr(default=None)
    # The label the tool's description shows the LLM for the declared file.
    _declared_label: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        file_path: str | None = None,
        base_dir: str | None = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        """Initialize the FileReadTool.

        Args:
            file_path (Optional[str]): Path to the file to be read. If provided,
                this becomes the default file path for the tool.
            base_dir (Optional[str]): Directory that runtime paths must stay
                inside. Defaults to the current working directory.
            encoding (str): Text encoding used to decode the file.
            **kwargs: Additional keyword arguments passed to BaseTool.
        """
        # Anchor base_dir once, so the sandbox root cannot move under a later
        # chdir while the declared file stays pinned to its original location.
        if base_dir is not None:
            base_dir = os.path.realpath(base_dir)

        display_path = None
        if file_path is not None:
            display_path = format_path_for_display(file_path, base_dir)
            kwargs["description"] = (
                f"A tool that reads file content. The default file is {display_path}, which is read when 'file_path' is omitted. You can also provide a different 'file_path' parameter to read another file, though reads are confined to the tool's allowed directory and a path that resolves outside it is rejected. Specify 'start_line' and 'line_count' to read specific parts of the file."
            )

        super().__init__(**kwargs)
        self.file_path = file_path
        self.base_dir = base_dir
        self.encoding = encoding
        self._declared_realpath = (
            _resolve_against_base(file_path, base_dir)
            if file_path is not None
            else None
        )
        self._declared_label = display_path

    def _resolve_path(self, file_path: str) -> str:
        """Resolve *file_path* and confirm the tool is allowed to read it.

        The file declared at construction time is always allowed: the developer
        named it, and the tool reads it anyway when ``file_path`` is omitted. It
        is addressable both by its real path and by the label the description
        shows the LLM, since that label is all the model is given. Everything
        else — including any path an LLM invents at runtime — must resolve
        inside ``base_dir``.

        Args:
            file_path: The path to resolve.

        Returns:
            The resolved, allowed absolute path.

        Raises:
            ValueError: If the path resolves outside ``base_dir``.
        """
        declared = self._declared_realpath
        if declared is not None and (
            file_path == self._declared_label
            or _resolve_against_base(file_path, self.base_dir) == declared
        ):
            return declared
        return validate_file_path(file_path, self.base_dir)

    def _run(
        self,
        file_path: str | None = None,
        start_line: int | None = 1,
        line_count: int | None = None,
    ) -> str:
        """Read a file, or a window of its lines, as text."""
        start_line = start_line or 1
        line_count = line_count or None

        if file_path is None:
            if self._declared_realpath is None:
                return "Error: No file path provided. Please provide a file path either in the constructor or as an argument."
            file_path = self._declared_realpath
        else:
            try:
                file_path = self._resolve_path(file_path)
            except ValueError as e:
                return "Error: Invalid file path: " + format_sandbox_error(
                    e,
                    "Pass base_dir to FileReadTool to allow reading another "
                    "directory tree.",
                )

        display_path = format_path_for_display(file_path, self.base_dir)
        try:
            with open(file_path, "r", encoding=self.encoding) as file:
                if start_line == 1 and line_count is None:
                    return file.read()

                start_idx = max(start_line - 1, 0)
                stop_idx = None if line_count is None else start_idx + line_count

                # islice stops pulling lines once stop_idx is reached, so a small
                # window near the top of a huge file does not scan the whole file.
                selected_lines = list(islice(file, start_idx, stop_idx))

                if not selected_lines and start_idx > 0:
                    return f"Error: Start line {start_line} exceeds the number of lines in the file."

                return "".join(selected_lines)
        except FileNotFoundError:
            return f"Error: File not found at path: {display_path}"
        except PermissionError:
            return f"Error: Permission denied when trying to read file: {display_path}"
        except UnicodeDecodeError:
            return (
                f"Error: Failed to decode file {display_path} as {self.encoding}. "
                f"Pass a different 'encoding' to FileReadTool if the file uses "
                f"another text encoding."
            )
        except Exception as e:
            return (
                f"Error: Failed to read file {display_path}. "
                f"{format_error_for_display(e)}"
            )
