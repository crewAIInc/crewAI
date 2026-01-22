"""Enums for file processing configuration."""

from enum import Enum


class FileHandling(Enum):
    """Defines how files exceeding provider limits should be handled.

    Attributes:
        STRICT: Fail with an error if file exceeds limits.
        AUTO: Automatically resize, compress, or optimize to fit limits.
        WARN: Log a warning but attempt to process anyway.
        CHUNK: Split large files into smaller pieces.
    """

    STRICT = "strict"
    AUTO = "auto"
    WARN = "warn"
    CHUNK = "chunk"
