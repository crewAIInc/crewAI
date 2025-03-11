from enum import Enum
from typing import Any, Dict, List, Optional, Union

from crewai.utilities.events.base_events import CrewEvent


class LLMCallType(Enum):
    """Type of LLM call being made"""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"


class LLMCallStartedEvent(CrewEvent):
    """Event emitted when a LLM call starts"""

    type: str = "llm_call_started"
    messages: Union[str, List[Dict[str, str]]]
    tools: Optional[List[dict]] = None
    callbacks: Optional[List[Any]] = None
    available_functions: Optional[Dict[str, Any]] = None


class LLMCallCompletedEvent(CrewEvent):
    """Event emitted when a LLM call completes"""

    type: str = "llm_call_completed"
    response: Any
    call_type: LLMCallType


class LLMCallFailedEvent(CrewEvent):
    """Event emitted when a LLM call fails"""

    error: str
    type: str = "llm_call_failed"


class LLMStreamChunkEvent(CrewEvent):
    """Event emitted when a streaming chunk is received"""

    type: str = "llm_stream_chunk"
    chunk: str
