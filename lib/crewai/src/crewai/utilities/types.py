"""Types for CrewAI utilities."""

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict


class LLMMessage(TypedDict):
    """Type for formatted LLM messages.

    Notes:
        - TODO: Update the LLM.call & BaseLLM.call signatures to use this type
          instead of str | list[dict[str, str]]
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict[str, Any]] | None
    tool_call_id: NotRequired[str]
    name: NotRequired[str]
    tool_calls: NotRequired[list[dict[str, Any]]]
