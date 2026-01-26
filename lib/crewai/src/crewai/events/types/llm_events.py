from enum import Enum
from typing import Any

from pydantic import BaseModel

from crewai.events.base_events import BaseEvent


class LLMEventBase(BaseEvent):
    from_task: Any | None = None
    from_agent: Any | None = None
    model: str | None = None

    def __init__(self, **data: Any) -> None:
        if data.get("from_task"):
            task = data["from_task"]
            data["task_id"] = str(task.id)
            data["task_name"] = task.name or task.description
            data["from_task"] = None

        if data.get("from_agent"):
            agent = data["from_agent"]
            data["agent_id"] = str(agent.id)
            data["agent_role"] = agent.role
            data["from_agent"] = None

        super().__init__(**data)


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
    call_type: LLMCallType | None = None
    response_id: str | None = None
