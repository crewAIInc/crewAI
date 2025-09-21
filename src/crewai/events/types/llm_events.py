from enum import Enum
from typing import Any

from pydantic import BaseModel

from crewai.events.base_events import BaseEvent


class LLMEventBase(BaseEvent):
    task_name: str | None = None
    task_id: str | None = None

    agent_id: str | None = None
    agent_role: str | None = None

    from_task: Any | None = None
    from_agent: Any | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_agent_params(data)
        self._set_task_params(data)


class LLMCallType(Enum):
    """Type of LLM call being made"""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"


class LLMCallStartedEvent(LLMEventBase):
    """Event emitted when a LLM call starts

    Attributes:
        messages: Content can be either a string or a list of dictionaries that support
            multimodal content (text, images, etc.)
    """

    type: str = "llm_call_started"
    model: str | None = None
    messages: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    callbacks: list[Any] | None = None
    available_functions: dict[str, Any] | None = None


class LLMCallCompletedEvent(LLMEventBase):
    """Event emitted when a LLM call completes"""

    type: str = "llm_call_completed"
    messages: str | list[dict[str, Any]] | None = None
    response: Any
    call_type: LLMCallType
    model: str | None = None


class LLMCallFailedEvent(LLMEventBase):
    """Event emitted when a LLM call fails"""

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


class LLMStreamChunkEvent(LLMEventBase):
    """Event emitted when a streaming chunk is received"""

    type: str = "llm_stream_chunk"
    chunk: str
    tool_call: ToolCall | None = None
