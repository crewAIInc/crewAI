"""Types for CrewAI utilities."""

from typing import Any, Literal

from typing_extensions import TypedDict


MessageRole = Literal["user", "assistant", "system"]


class LLMMessage(TypedDict):
    """Type for formatted LLM messages.

    Notes:
        - TODO: Update the LLM.call & BaseLLM.call signatures to use this type
          instead of str | list[dict[str, str]]
    """

    role: MessageRole
    content: str | list[dict[str, Any]]
