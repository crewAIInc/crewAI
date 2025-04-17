from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from crewai.utilities.events.base_events import BaseEvent


class LLMCallType(Enum):
    """Type of LLM call being made"""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"


class LLMCallStartedEvent(BaseEvent):
    """Event emitted when a LLM call starts

    Attributes:
        messages: Content can be either a string or a list of dictionaries that support
            multimodal content (text, images, etc.)
    """

    type: str = "llm_call_started"
    messages: Union[str, List[Dict[str, Any]]]
    tools: Optional[List[dict]] = None
    callbacks: Optional[List[Any]] = None
    available_functions: Optional[Dict[str, Any]] = None


class LLMCallCompletedEvent(BaseEvent):
    """Event emitted when a LLM call completes"""

    type: str = "llm_call_completed"
    response: Any
    call_type: LLMCallType


class LLMCallFailedEvent(BaseEvent):
    """Event emitted when a LLM call fails"""

    error: str
    type: str = "llm_call_failed"


class FunctionCall(BaseModel):
    arguments: str
    name: Optional[str] = None


class ToolCall(BaseModel):
    id: Optional[str] = None
    function: FunctionCall
    type: Optional[str] = None
    index: int


class LLMStreamChunkEvent(BaseEvent):
    """Event emitted when a streaming chunk is received"""

    type: str = "llm_stream_chunk"
    chunk: str
    tool_call: Optional[ToolCall] = None
