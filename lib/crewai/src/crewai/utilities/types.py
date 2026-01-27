"""Types for CrewAI utilities."""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict


try:
    from crewai_files import FileInput
except ImportError:
    FileInput = Any  # type: ignore[misc,assignment]


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
    raw_tool_call_parts: NotRequired[list[Any]]
    files: NotRequired[dict[str, FileInput]]
