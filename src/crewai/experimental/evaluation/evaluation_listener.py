from collections.abc import Sequence
from datetime import datetime
from typing import Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolExecutionErrorEvent,
    ToolSelectionErrorEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolValidateInputErrorEvent,
)
from crewai.task import Task


class EvaluationTraceCallback(BaseEventListener):
    """Event listener for collecting execution traces for evaluation.

    This listener attaches to the event bus to collect detailed information
    about the execution process, including agent steps, tool uses, knowledge
    retrievals, and final output - all for use in agent evaluation.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized") or not self._initialized:
            super().__init__()
            self.traces = {}
            self.current_agent_id = None
            self.current_task_id = None
            self._initialized = True

    def setup_listeners(self, event_bus: CrewAIEventsBus):
        @event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event: AgentExecutionStartedEvent):
            self.on_agent_start(event.agent, event.task)

        @event_bus.on(LiteAgentExecutionStartedEvent)
        def on_lite_agent_started(source, event: LiteAgentExecutionStartedEvent):
            self.on_lite_agent_start(event.agent_info)

        @event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event: AgentExecutionCompletedEvent):
            self.on_agent_finish(event.agent, event.task, event.output)

        @event_bus.on(LiteAgentExecutionCompletedEvent)
        def on_lite_agent_completed(source, event: LiteAgentExecutionCompletedEvent):
            self.on_lite_agent_finish(event.output)

        @event_bus.on(ToolUsageFinishedEvent)
        def on_tool_completed(source, event: ToolUsageFinishedEvent):
            self.on_tool_use(
                event.tool_name, event.tool_args, event.output, success=True
            )

        @event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolUsageErrorEvent):
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="usage_error",
            )

        @event_bus.on(ToolExecutionErrorEvent)
        def on_tool_execution_error(source, event: ToolExecutionErrorEvent):
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="execution_error",
            )

        @event_bus.on(ToolSelectionErrorEvent)
        def on_tool_selection_error(source, event: ToolSelectionErrorEvent):
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="selection_error",
            )

        @event_bus.on(ToolValidateInputErrorEvent)
        def on_tool_validate_input_error(source, event: ToolValidateInputErrorEvent):
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="validation_error",
            )

        @event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(source, event: LLMCallStartedEvent):
            self.on_llm_call_start(event.messages, event.tools)

        @event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(source, event: LLMCallCompletedEvent):
            self.on_llm_call_end(event.messages, event.response)

    def on_lite_agent_start(self, agent_info: dict[str, Any]):
        self.current_agent_id = agent_info["id"]
        self.current_task_id = "lite_task"

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        self._init_trace(
            trace_key=trace_key,
            agent_id=self.current_agent_id,
            task_id=self.current_task_id,
            tool_uses=[],
            llm_calls=[],
            start_time=datetime.now(),
            final_output=None,
        )

    def _init_trace(self, trace_key: str, **kwargs: Any):
        self.traces[trace_key] = kwargs

    def on_agent_start(self, agent: BaseAgent, task: Task):
        self.current_agent_id = agent.id
        self.current_task_id = task.id

        trace_key = f"{agent.id}_{task.id}"
        self._init_trace(
            trace_key=trace_key,
            agent_id=agent.id,
            task_id=task.id,
            tool_uses=[],
            llm_calls=[],
            start_time=datetime.now(),
            final_output=None,
        )

    def on_agent_finish(self, agent: BaseAgent, task: Task, output: Any):
        trace_key = f"{agent.id}_{task.id}"
        if trace_key in self.traces:
            self.traces[trace_key]["final_output"] = output
            self.traces[trace_key]["end_time"] = datetime.now()

        self._reset_current()

    def _reset_current(self):
        self.current_agent_id = None
        self.current_task_id = None

    def on_lite_agent_finish(self, output: Any):
        trace_key = f"{self.current_agent_id}_lite_task"
        if trace_key in self.traces:
            self.traces[trace_key]["final_output"] = output
            self.traces[trace_key]["end_time"] = datetime.now()

        self._reset_current()

    def on_tool_use(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | str,
        result: Any,
        success: bool = True,
        error_type: str | None = None,
    ):
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key in self.traces:
            tool_use = {
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "success": success,
                "timestamp": datetime.now(),
            }

            # Add error information if applicable
            if not success and error_type:
                tool_use["error"] = True
                tool_use["error_type"] = error_type

            self.traces[trace_key]["tool_uses"].append(tool_use)

    def on_llm_call_start(
        self,
        messages: str | Sequence[dict[str, Any]] | None,
        tools: Sequence[dict[str, Any]] | None = None,
    ):
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key not in self.traces:
            return

        self.current_llm_call = {
            "messages": messages,
            "tools": tools,
            "start_time": datetime.now(),
            "response": None,
            "end_time": None,
        }

    def on_llm_call_end(
        self, messages: str | list[dict[str, Any]] | None, response: Any
    ):
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key not in self.traces:
            return

        total_tokens = 0
        if hasattr(response, "usage") and hasattr(response.usage, "total_tokens"):
            total_tokens = response.usage.total_tokens

        current_time = datetime.now()
        start_time = None
        if hasattr(self, "current_llm_call") and self.current_llm_call:
            start_time = self.current_llm_call.get("start_time")

        if not start_time:
            start_time = current_time
        llm_call = {
            "messages": messages,
            "response": response,
            "start_time": start_time,
            "end_time": current_time,
            "total_tokens": total_tokens,
        }

        self.traces[trace_key]["llm_calls"].append(llm_call)

        if hasattr(self, "current_llm_call"):
            self.current_llm_call = {}

    def get_trace(self, agent_id: str, task_id: str) -> dict[str, Any] | None:
        trace_key = f"{agent_id}_{task_id}"
        return self.traces.get(trace_key)


def create_evaluation_callbacks() -> EvaluationTraceCallback:
    from crewai.events.event_bus import crewai_event_bus

    callback = EvaluationTraceCallback()
    callback.setup_listeners(crewai_event_bus)
    return callback
