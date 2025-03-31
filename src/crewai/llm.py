from typing import Any, Dict, List, Optional, Union

from crewai.llms.base_llm import BaseLLM
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)

from langchain.chat_models import AzureChatOpenAI

class LLM(BaseLLM):
    def __init__(
        self,
        model: AzureChatOpenAI,
    ):
        self.model = model

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        # --- Emit call started event
        crewai_event_bus.emit(
            self,
            event=LLMCallStartedEvent(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
            ),
        )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        try:
            response = self.model(messages)
            content = response.content if hasattr(response, 'content') else response

            crewai_event_bus.emit(
                self,
                event=LLMCallCompletedEvent(
                    response=content, call_type=LLMCallType.LLM_CALL
                ),
            )
            return content

        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=LLMCallFailedEvent(error=str(e)),
            )
            raise