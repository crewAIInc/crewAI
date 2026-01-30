"""Utility functions for file handling."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from crewai_files.core.sources import is_file_source


if TYPE_CHECKING:
    from crewai_files.core.sources import FileSource, FileSourceInput
    from crewai_files.core.types import FileInput


__all__ = ["is_file_source", "normalize_input_files", "wrap_file_source"]


def wrap_file_source(source: FileSource) -> FileInput:
    """Wrap a FileSource in the appropriate typed FileInput wrapper.

    Args:
        source: The file source to wrap.

    Returns:
        Typed FileInput wrapper based on content type.
    """
    from crewai_files.core.types import (
        AudioFile,
        ImageFile,
        PDFFile,
        TextFile,
        VideoFile,
    )

    content_type = source.content_type

    if content_type.startswith("image/"):
        return ImageFile(source=source)
    if content_type.startswith("audio/"):
        return AudioFile(source=source)
    if content_type.startswith("video/"):
        return VideoFile(source=source)
    if content_type == "application/pdf":
        return PDFFile(source=source)
    return TextFile(source=source)


def normalize_input_files(
    input_files: list[FileSourceInput | FileInput],
) -> dict[str, FileInput]:
    """Convert a list of file sources to a named dictionary of FileInputs.

    Args:
        input_files: List of file source inputs or File objects.

    Returns:
        Dictionary mapping names to FileInput wrappers.
    """
    from crewai_files.core.sources import FileBytes, FilePath, FileStream, FileUrl
    from crewai_files.core.types import BaseFile

    result: dict[str, FileInput] = {}

    for i, item in enumerate(input_files):
        if isinstance(item, BaseFile):
            name = item.filename or f"file_{i}"
            if "." in name:
                name = name.rsplit(".", 1)[0]
            result[name] = item
            continue

        file_source: FilePath | FileBytes | FileStream | FileUrl
        if isinstance(item, (FilePath, FileBytes, FileStream, FileUrl)):
            file_source = item
        elif isinstance(item, Path):
            file_source = FilePath(path=item)
        elif isinstance(item, str):
            if item.startswith(("http://", "https://")):
                file_source = FileUrl(url=item)
            else:
                file_source = FilePath(path=Path(item))
        elif isinstance(item, (bytes, memoryview)):
            file_source = FileBytes(data=bytes(item))
        else:
            continue

        name = file_source.filename or f"file_{i}"
        result[name] = wrap_file_source(file_source)

    return result
