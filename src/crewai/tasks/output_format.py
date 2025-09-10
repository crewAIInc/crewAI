"""Task output format definitions for CrewAI."""

from enum import Enum


class OutputFormat(str, Enum):
    """Enum that represents the output format of a task.

    Attributes:
        JSON: Output as JSON dictionary format
        PYDANTIC: Output as Pydantic model instance
        RAW: Output as raw unprocessed string
    """

    JSON = "json"
    PYDANTIC = "pydantic"
    RAW = "raw"
