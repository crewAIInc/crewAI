from pydantic import InstanceOf


from typing import Any, List, Optional, Union

from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.utilities import I18N
from crewai.agents.tools_handler import ToolsHandler
from llama_index_client import ChatMessage
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    ChatResponseMode,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import (
    CBEventType,
    EventPayload,
)


class CrewLlamaReActAgentExecutor(ReActAgent):
    _i18n: I18N = I18N()
    should_ask_for_human_input: bool = False
    llm: Any = None
    iterations: int = 0
    task: Any = None
    original_tools: List[Any] = []
    crew_agent: Any = None
    crew: Any = None
    function_calling_llm: Any = None
    request_within_rpm_limit: Any = None
    tools_handler: Optional[InstanceOf[ToolsHandler]] = None
    max_iterations: Optional[int] = 15
    have_forced_answer: bool = False
    force_answer_max_iterations: Optional[int] = None
    step_callback: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(
            callback_manager=kwargs.get("callback_manager"),
            llm=kwargs.get("llm"),
            verbose=kwargs.get("verbose", False),
            memory=kwargs.get("memory"),
            tools=kwargs.get("tools"),
            max_iterations=kwargs.get("max_iterations", 25),
        )

    def _create_short_term_memory(self, output) -> None:
        if (
            self.crew
            and self.crew.memory
            and "Action: Delegate work to coworker" not in output.log
        ):
            memory = ShortTermMemoryItem(
                data=output.log,
                agent=self.crew_agent.role,
                metadata={
                    "observation": self.task.description,
                },
            )
            self.crew._short_term_memory.save(memory)

    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> AgentChatResponse:
        # override tool choice is provided as input.
        if tool_choice is None:
            tool_choice = self.default_tool_choice
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message=message,
                chat_history=chat_history,
                tool_choice=tool_choice,
                mode=ChatResponseMode.WAIT,
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
            # TODO: Add logic for handling human input when e.finished
            if e._event_type == CBEventType.AGENT_STEP:
                self.iterations += 1
            if e.finished:
                self._create_short_term_memory(chat_response)

        return chat_response
