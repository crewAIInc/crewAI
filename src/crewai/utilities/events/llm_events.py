from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from crewai.utilities.events.base_events import CrewEvent


class LLMCallType(Enum):
    """Type of LLM call being made"""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"


class ContentType(str, Enum):
    """Types of content in multimodal messages"""
    
    TEXT = "text"
    IMAGE_URL = "image_url"


class LLMCallStartedEvent(CrewEvent):
    """Event emitted when a LLM call starts"""

    type: str = "llm_call_started"
    messages: Union[
        str,
        List[Union[
            str,
            Dict[str, Union[
                str,
                List[Dict[str, Union[
                    str,
                    Dict[Literal["url"], str]
                ]]]
            ]]
        ]]
    ]
    """
    Supports both string messages and structured messages including multimodal content.
    Formats supported:
    1. Simple string: "This is a message"
    2. List of message objects: [{"role": "user", "content": "Hello"}]
    3. Mixed list with strings and objects: ["Simple message", {"role": "user", "content": "Hello"}]
    4. Multimodal format:
    {
        'role': str,
        'content': List[
            Union[
                Dict[Literal["type", "text"], str],
                Dict[Literal["type", "image_url"], Dict[str, str]]
            ]
        ]
    }
    """
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
