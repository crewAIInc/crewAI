from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from crewai.events.base_events import BaseEvent


class LLMEventBase(BaseEvent):
    from_task: Any | None = None
    from_agent: Any | None = None
    model: str | None = None
    call_id: str

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

    type: Literal["llm_call_started"] = "llm_call_started"
    messages: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    callbacks: list[Any] | None = None
    available_functions: dict[str, Any] | None = None
    # Sampling/request parameters forwarded for OTel GenAI compliance.
    # All optional so legacy emitters keep working unchanged.
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    n: int | None = None


class LLMCallCompletedEvent(LLMEventBase):
    """Event emitted when a LLM call completes"""

    type: Literal["llm_call_completed"] = "llm_call_completed"
    messages: str | list[dict[str, Any]] | None = None
    response: Any
    call_type: LLMCallType
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    response_id: str | None = None

    @field_validator("finish_reason", "response_id", mode="before")
    @classmethod
    def _coerce_non_string_to_none(cls, value: Any) -> str | None:
        """Drop non-string values so test mocks and exotic provider types
        (MagicMock, protobuf enums, etc.) never crash event construction.

        Provider helpers are best-effort: when extraction returns something
        non-string (e.g. a ``MagicMock`` in unit tests), we treat it as
        "no value" rather than raising. Downstream telemetry already
        handles the missing-attribute case.
        """
        if value is None or isinstance(value, str):
            return value
        return None


class LLMCallFailedEvent(LLMEventBase):
    """Event emitted when a LLM call fails"""

    error: str
    type: Literal["llm_call_failed"] = "llm_call_failed"


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

    type: Literal["llm_stream_chunk"] = "llm_stream_chunk"
    chunk: str
    tool_call: ToolCall | None = None
    call_type: LLMCallType | None = None
    response_id: str | None = None


class LLMThinkingChunkEvent(LLMEventBase):
    """Event emitted when a thinking/reasoning chunk is received from a thinking model"""

    type: Literal["llm_thinking_chunk"] = "llm_thinking_chunk"
    chunk: str
    response_id: str | None = None
