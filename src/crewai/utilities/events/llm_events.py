import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, model_validator

from crewai.utilities.events.base_events import BaseEvent

logger = logging.getLogger(__name__)


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

    @model_validator(mode='before')
    @classmethod
    def sanitize_tools(cls: Type["LLMCallStartedEvent"], values: Any) -> Any:
        """Sanitize tools list to only include dict objects, filtering out non-dict objects like TokenCalcHandler.
        
        Args:
            values (dict): Input values dictionary containing tools and other event data.

        Returns:
            dict: Sanitized values with filtered tools list containing only valid dict objects.
        
        Example:
            >>> from crewai.utilities.token_counter_callback import TokenCalcHandler
            >>> from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
            >>> token_handler = TokenCalcHandler(TokenProcess())
            >>> tools = [{"name": "tool1"}, token_handler, {"name": "tool2"}]
            >>> sanitized = cls.sanitize_tools({"tools": tools})
            >>> # Expected: {"tools": [{"name": "tool1"}, {"name": "tool2"}]}
        """
        try:
            if isinstance(values, dict) and 'tools' in values and values['tools'] is not None:
                if isinstance(values['tools'], list):
                    sanitized_tools = []
                    for tool in values['tools']:
                        if isinstance(tool, dict):
                            if all(isinstance(v, (str, int, float, bool, dict, list, type(None))) for v in tool.values()):
                                sanitized_tools.append(tool)
                            else:
                                logger.warning(f"Tool dict contains invalid value types: {tool}")
                        else:
                            logger.debug(f"Filtering out non-dict tool object: {type(tool).__name__}")
                    
                    values['tools'] = sanitized_tools
        except Exception as e:
            logger.warning(f"Error during tools sanitization: {e}")
        
        return values


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
