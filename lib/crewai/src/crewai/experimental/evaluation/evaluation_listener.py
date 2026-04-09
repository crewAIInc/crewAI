"""Event listener for collecting execution traces for evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

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

    _instance: EvaluationTraceCallback | None = None
    _initialized: bool = False

    def __new__(cls) -> EvaluationTraceCallback:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the evaluation trace callback."""
        if not self._initialized:
            super().__init__()
            self.traces: dict[str, Any] = {}
            self.current_agent_id: UUID | str | None = None
            self.current_task_id: UUID | str | None = None
            self.current_llm_call: dict[str, Any] = {}
            self._initialized = True

    def setup_listeners(self, event_bus: CrewAIEventsBus) -> None:
        """Set up event listeners on the event bus.

        Args:
            event_bus: The event bus to register listeners on.
        """

        @event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source: Any, event: AgentExecutionStartedEvent) -> None:
            self.on_agent_start(event.agent, event.task)

        @event_bus.on(LiteAgentExecutionStartedEvent)
        def on_lite_agent_started(
            source: Any, event: LiteAgentExecutionStartedEvent
        ) -> None:
            self.on_lite_agent_start(event.agent_info)

        @event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(
            source: Any, event: AgentExecutionCompletedEvent
        ) -> None:
            self.on_agent_finish(event.agent, event.task, event.output)

        @event_bus.on(LiteAgentExecutionCompletedEvent)
        def on_lite_agent_completed(
            source: Any, event: LiteAgentExecutionCompletedEvent
        ) -> None:
            self.on_lite_agent_finish(event.output)

        @event_bus.on(ToolUsageFinishedEvent)
        def on_tool_completed(source: Any, event: ToolUsageFinishedEvent) -> None:
            self.on_tool_use(
                event.tool_name, event.tool_args, event.output, success=True
            )

        @event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source: Any, event: ToolUsageErrorEvent) -> None:
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="usage_error",
            )

        @event_bus.on(ToolExecutionErrorEvent)
        def on_tool_execution_error(
            source: Any, event: ToolExecutionErrorEvent
        ) -> None:
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="execution_error",
            )

        @event_bus.on(ToolSelectionErrorEvent)
        def on_tool_selection_error(
            source: Any, event: ToolSelectionErrorEvent
        ) -> None:
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="selection_error",
            )

        @event_bus.on(ToolValidateInputErrorEvent)
        def on_tool_validate_input_error(
            source: Any, event: ToolValidateInputErrorEvent
        ) -> None:
            self.on_tool_use(
                event.tool_name,
                event.tool_args,
                event.error,
                success=False,
                error_type="validation_error",
            )

        @event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(source: Any, event: LLMCallStartedEvent) -> None:
            self.on_llm_call_start(event.messages, event.tools)

        @event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(source: Any, event: LLMCallCompletedEvent) -> None:
            self.on_llm_call_end(event.messages, event.response)

    def on_lite_agent_start(self, agent_info: dict[str, Any]) -> None:
        """Handle a lite agent execution start event.

        Args:
            agent_info: Dictionary containing agent information.
        """
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

    def _init_trace(self, trace_key: str, **kwargs: Any) -> None:
        """Initialize a trace entry.

        Args:
            trace_key: The key to store the trace under.
            **kwargs: Trace metadata to store.
        """
        self.traces[trace_key] = kwargs

    def on_agent_start(self, agent: BaseAgent, task: Task) -> None:
        """Handle an agent execution start event.

        Args:
            agent: The agent that started execution.
            task: The task being executed.
        """
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

    def on_agent_finish(self, agent: BaseAgent, task: Task, output: Any) -> None:
        """Handle an agent execution completion event.

        Args:
            agent: The agent that finished execution.
            task: The task that was executed.
            output: The agent's output.
        """
        trace_key = f"{agent.id}_{task.id}"
        if trace_key in self.traces:
            self.traces[trace_key]["final_output"] = output
            self.traces[trace_key]["end_time"] = datetime.now()

        self._reset_current()

    def _reset_current(self) -> None:
        """Reset the current agent and task tracking state."""
        self.current_agent_id = None
        self.current_task_id = None

    def on_lite_agent_finish(self, output: Any) -> None:
        """Handle a lite agent execution completion event.

        Args:
            output: The agent's output.
        """
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
    ) -> None:
        """Record a tool usage event in the current trace.

        Args:
            tool_name: Name of the tool used.
            tool_args: Arguments passed to the tool.
            result: The tool's output or error message.
            success: Whether the tool call succeeded.
            error_type: Type of error if the call failed.
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key in self.traces:
            tool_use: dict[str, Any] = {
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "success": success,
                "timestamp": datetime.now(),
            }

            if not success and error_type:
                tool_use["error"] = True
                tool_use["error_type"] = error_type

            self.traces[trace_key]["tool_uses"].append(tool_use)

    def on_llm_call_start(
        self,
        messages: str | Sequence[dict[str, Any]] | None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Record an LLM call start event.

        Args:
            messages: The messages sent to the LLM.
            tools: Tool definitions provided to the LLM.
        """
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
    ) -> None:
        """Record an LLM call completion event.

        Args:
            messages: The messages from the LLM call.
            response: The LLM response object.
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key not in self.traces:
            return

        total_tokens = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            total_tokens = getattr(usage, "total_tokens", 0)

        current_time = datetime.now()
        start_time = (
            self.current_llm_call.get("start_time") if self.current_llm_call else None
        )

        if not start_time:
            start_time = current_time
        llm_call: dict[str, Any] = {
            "messages": messages,
            "response": response,
            "start_time": start_time,
            "end_time": current_time,
            "total_tokens": total_tokens,
        }

        self.traces[trace_key]["llm_calls"].append(llm_call)
        self.current_llm_call = {}

    def get_trace(self, agent_id: str, task_id: str) -> dict[str, Any] | None:
        """Retrieve a trace by agent and task ID.

        Args:
            agent_id: The agent's identifier.
            task_id: The task's identifier.

        Returns:
            The trace dictionary, or None if not found.
        """
        trace_key = f"{agent_id}_{task_id}"
        return self.traces.get(trace_key)


def create_evaluation_callbacks() -> EvaluationTraceCallback:
    """Create and register an evaluation trace callback on the event bus.

    Returns:
        The configured EvaluationTraceCallback instance.
    """
    from crewai.events.event_bus import crewai_event_bus

    callback = EvaluationTraceCallback()
    callback.setup_listeners(crewai_event_bus)
    return callback
