"""Types for CrewAI utilities."""

from typing import Any, Literal, TypedDict

from crewai_files import FileInput


class LLMMessage(TypedDict):
    """Type for formatted LLM messages.

    Notes:
        - TODO: Update the LLM.call & BaseLLM.call signatures to use this type
          instead of str | list[dict[str, str]]
    """

    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]]


class KickoffInputs(TypedDict, total=False):
    """Type for crew kickoff inputs.

    Attributes:
        files: Named file inputs accessible to tasks during execution.
    """

    files: dict[str, FileInput]
