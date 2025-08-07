from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from crewai.utilities.events.base_events import BaseEvent


class LLMEventBase(BaseEvent):
    task_name: Optional[str] = None
    task_id: Optional[str] = None

    agent_id: Optional[str] = None
    agent_role: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_agent_params(data)
        self._set_task_params(data)

    def _set_agent_params(self, data: Dict[str, Any]):
        task = data.get("from_task", None)
        agent = task.agent if task else data.get("from_agent", None)

        if not agent:
            return

        self.agent_id = agent.id
        self.agent_role = agent.role

    def _set_task_params(self, data: Dict[str, Any]):
        if "from_task" in data and (task := data["from_task"]):
            self.task_id = task.id
            self.task_name = task.name


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
    model: Optional[str] = None
    messages: Optional[Union[str, List[Dict[str, Any]]]] = None
    tools: Optional[List[dict[str, Any]]] = None
    callbacks: Optional[List[Any]] = None
    available_functions: Optional[Dict[str, Any]] = None


class LLMCallCompletedEvent(LLMEventBase):
    """Event emitted when a LLM call completes"""

    type: str = "llm_call_completed"
    messages: str | list[dict[str, Any]] | None = None
    response: Any
    call_type: LLMCallType
    model: Optional[str] = None


class LLMCallFailedEvent(LLMEventBase):
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


class LLMStreamChunkEvent(LLMEventBase):
    """Event emitted when a streaming chunk is received"""

    type: str = "llm_stream_chunk"
    chunk: str
    tool_call: Optional[ToolCall] = None
