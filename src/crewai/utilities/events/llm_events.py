from enum import Enum
from typing import Any

from pydantic import BaseModel

from crewai.utilities.events.base_events import BaseEvent


class LLMCallType(Enum):
    """Type of LLM call being made."""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"


class LLMCallStartedEvent(BaseEvent):
    """Event emitted when a LLM call starts.

    Attributes:
        messages: Content can be either a string or a list of dictionaries that support
            multimodal content (text, images, etc.)

    """

    type: str = "llm_call_started"
    messages: str | list[dict[str, Any]]
    tools: list[dict] | None = None
    callbacks: list[Any] | None = None
    available_functions: dict[str, Any] | None = None


class LLMCallCompletedEvent(BaseEvent):
    """Event emitted when a LLM call completes."""

    type: str = "llm_call_completed"
    response: Any
    call_type: LLMCallType


class LLMCallFailedEvent(BaseEvent):
    """Event emitted when a LLM call fails."""

    error: str
    type: str = "llm_call_failed"


class FunctionCall(BaseModel):
    arguments: str
    name: str | None = None


class ToolCall(BaseModel):
    id: str | None = None
    function: FunctionCall
    type: str | None = None
    index: int


class LLMStreamChunkEvent(BaseEvent):
    """Event emitted when a streaming chunk is received."""

    type: str = "llm_stream_chunk"
    chunk: str
    tool_call: ToolCall | None = None
