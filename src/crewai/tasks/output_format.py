from enum import Enum


class OutputFormat(str, Enum):
    """Enum that represents the output format of a task."""

    JSON = "json"
    PYDANTIC = "pydantic"
    RAW = "raw"
