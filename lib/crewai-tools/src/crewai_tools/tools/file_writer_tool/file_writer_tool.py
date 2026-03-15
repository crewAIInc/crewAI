import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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


class ScopedFileWriterToolInput(BaseModel):
    """Input when base_dir is set — the LLM supplies only filename and content."""
    filename: str
    overwrite: str | bool = False
    content: str


class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = (
        "A tool to write content to a specified file. "
        "Accepts filename, content, and optionally a directory path and overwrite flag as input."
    )
    args_schema: type[BaseModel] = FileWriterToolInput
    base_dir: str | None = None

    def __init__(self, base_dir: str | None = None, **kwargs: Any) -> None:
        """Initialize the FileWriterTool.

        Args:
            base_dir (Optional[str]): Restrict all writes to this directory tree.
                Any filename or directory that resolves outside base_dir is rejected,
                including ../traversal and symlink escapes. When not set the tool
                can write anywhere the process has permission to — only use that in
                fully sandboxed environments.
        """
        super().__init__(**kwargs)
        self.base_dir = os.path.realpath(base_dir) if base_dir is not None else None
        if base_dir is not None:
            self.args_schema = ScopedFileWriterToolInput
            self.description = (
                f"A tool to write files into {base_dir}. "
                "Accepts a filename, content, and an optional overwrite flag."
            )
            self._generate_description()

    def _validate_path(self, filepath: str) -> str | None:
        """Resolve path and enforce base_dir containment. Returns None if rejected."""
        if self.base_dir is None:
            return filepath
        real_path = os.path.realpath(filepath)
        if not real_path.startswith(self.base_dir + os.sep) and real_path != self.base_dir:
            return None
        return real_path

    def _run(self, **kwargs: Any) -> str:
        try:
            filename = kwargs["filename"]

            if self.base_dir is not None:
                # Developer controls the directory; LLM only supplies filename.
                filepath = os.path.join(self.base_dir, filename)
            else:
                directory = kwargs.get("directory") or "./"
                filepath = os.path.join(directory, filename)

            validated = self._validate_path(filepath)
            if validated is None:
                return "Error: Access denied — path resolves outside the allowed directory."

            validated_dir = os.path.dirname(validated)
            os.makedirs(validated_dir, exist_ok=True)

            kwargs["overwrite"] = strtobool(kwargs["overwrite"])

            if os.path.exists(validated) and not kwargs["overwrite"]:
                return f"File {validated} already exists and overwrite option was not passed."

            mode = "w" if kwargs["overwrite"] else "x"
            with open(validated, mode) as file:
                file.write(kwargs["content"])
            return f"Content successfully written to {validated}"
        except FileExistsError:
            return f"File already exists and overwrite option was not passed."
        except KeyError as e:
            return f"An error occurred while accessing key: {e!s}"
        except Exception as e:
            return f"An error occurred while writing to the file: {e!s}"
