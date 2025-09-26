from __future__ import annotations

from io import StringIO
from typing import Any

from pydantic import Field, PrivateAttr

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTestResultEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    CrewTrainStartedEvent,
)
from crewai.events.types.knowledge_events import (
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeRetrievalStartedEvent,
    KnowledgeSearchQueryFailedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
)
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.events.types.logging_events import (
    AgentLogsExecutionEvent,
    AgentLogsStartedEvent,
)
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.llm import LLM
from crewai.task import Task
from crewai.telemetry.telemetry import Telemetry
from crewai.utilities import Logger
from crewai.utilities.constants import EMITTER_COLOR

from .listeners.memory_listener import MemoryListener
from .types.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from .types.reasoning_events import (
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
)
from .types.task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from .types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)


class EventListener(BaseEventListener):
    _instance = None
    _telemetry: Telemetry = PrivateAttr(default_factory=lambda: Telemetry())
    logger = Logger(verbose=True, default_color=EMITTER_COLOR)
    execution_spans: dict[Task, Any] = Field(default_factory=dict)
    next_chunk = 0
    text_stream = StringIO()
    knowledge_retrieval_in_progress = False
    knowledge_query_in_progress = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized") or not self._initialized:
            super().__init__()
            self._telemetry = Telemetry()
            self._telemetry.set_tracer()
            self.execution_spans = {}
            self._initialized = True
            self.formatter = ConsoleFormatter(verbose=True)

            MemoryListener(formatter=self.formatter)

    # ----------- CREW EVENTS -----------

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            self.formatter.create_crew_tree(event.crew_name or "Crew", source.id)
            self._telemetry.crew_execution_span(source, event.inputs)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            # Handle telemetry
            final_string_output = event.output.raw
            self._telemetry.end_crew(source, final_string_output)

            self.formatter.update_crew_tree(
                self.formatter.current_crew_tree,
                event.crew_name or "Crew",
                source.id,
                "completed",
                final_string_output,
            )

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event: CrewKickoffFailedEvent):
            self.formatter.update_crew_tree(
                self.formatter.current_crew_tree,
                event.crew_name or "Crew",
                source.id,
                "failed",
            )

        @crewai_event_bus.on(CrewTrainStartedEvent)
        def on_crew_train_started(source, event: CrewTrainStartedEvent):
            self.formatter.handle_crew_train_started(
                event.crew_name or "Crew", str(event.timestamp)
            )

        @crewai_event_bus.on(CrewTrainCompletedEvent)
        def on_crew_train_completed(source, event: CrewTrainCompletedEvent):
            self.formatter.handle_crew_train_completed(
                event.crew_name or "Crew", str(event.timestamp)
            )

        @crewai_event_bus.on(CrewTrainFailedEvent)
        def on_crew_train_failed(source, event: CrewTrainFailedEvent):
            self.formatter.handle_crew_train_failed(event.crew_name or "Crew")

        @crewai_event_bus.on(CrewTestResultEvent)
        def on_crew_test_result(source, event: CrewTestResultEvent):
            self._telemetry.individual_test_result_span(
                source.crew,
                event.quality,
                int(event.execution_duration),
                event.model,
            )

        # ----------- TASK EVENTS -----------

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source, event: TaskStartedEvent):
            span = self._telemetry.task_started(crew=source.agent.crew, task=source)
            self.execution_spans[source] = span
            # Pass both task ID and task name (if set)
            task_name = source.name if hasattr(source, "name") and source.name else None
            self.formatter.create_task_branch(
                self.formatter.current_crew_tree, source.id, task_name
            )

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event: TaskCompletedEvent):
            # Handle telemetry
            span = self.execution_spans.get(source)
            if span:
                self._telemetry.task_ended(span, source, source.agent.crew)
            self.execution_spans[source] = None

            # Pass task name if it exists
            task_name = source.name if hasattr(source, "name") and source.name else None
            self.formatter.update_task_status(
                self.formatter.current_crew_tree,
                source.id,
                source.agent.role,
                "completed",
                task_name,
            )

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event: TaskFailedEvent):
            span = self.execution_spans.get(source)
            if span:
                if source.agent and source.agent.crew:
                    self._telemetry.task_ended(span, source, source.agent.crew)
                self.execution_spans[source] = None

            # Pass task name if it exists
            task_name = source.name if hasattr(source, "name") and source.name else None
            self.formatter.update_task_status(
                self.formatter.current_crew_tree,
                source.id,
                source.agent.role,
                "failed",
                task_name,
            )

        # ----------- AGENT EVENTS -----------

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            self.formatter.create_agent_branch(
                self.formatter.current_task_branch,
                event.agent.role,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            self.formatter.update_agent_status(
                self.formatter.current_agent_branch,
                event.agent.role,
                self.formatter.current_crew_tree,
            )

        # ----------- LITE AGENT EVENTS -----------

        @crewai_event_bus.on(LiteAgentExecutionStartedEvent)
        def on_lite_agent_execution_started(
            source, event: LiteAgentExecutionStartedEvent
        ):
            """Handle LiteAgent execution started event."""
            self.formatter.handle_lite_agent_execution(
                event.agent_info["role"], status="started", **event.agent_info
            )

        @crewai_event_bus.on(LiteAgentExecutionCompletedEvent)
        def on_lite_agent_execution_completed(
            source, event: LiteAgentExecutionCompletedEvent
        ):
            """Handle LiteAgent execution completed event."""
            self.formatter.handle_lite_agent_execution(
                event.agent_info["role"], status="completed", **event.agent_info
            )

        @crewai_event_bus.on(LiteAgentExecutionErrorEvent)
        def on_lite_agent_execution_error(source, event: LiteAgentExecutionErrorEvent):
            """Handle LiteAgent execution error event."""
            self.formatter.handle_lite_agent_execution(
                event.agent_info["role"],
                status="failed",
                error=event.error,
                **event.agent_info,
            )

        # ----------- FLOW EVENTS -----------

        @crewai_event_bus.on(FlowCreatedEvent)
        def on_flow_created(source, event: FlowCreatedEvent):
            self._telemetry.flow_creation_span(event.flow_name)
            self.formatter.create_flow_tree(event.flow_name, str(source.flow_id))

        @crewai_event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event: FlowStartedEvent):
            self._telemetry.flow_execution_span(
                event.flow_name, list(source._methods.keys())
            )
            self.formatter.start_flow(event.flow_name, str(source.flow_id))

        @crewai_event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event: FlowFinishedEvent):
            self.formatter.update_flow_status(
                self.formatter.current_flow_tree, event.flow_name, source.flow_id
            )

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(source, event: MethodExecutionStartedEvent):
            self.formatter.update_method_status(
                self.formatter.current_method_branch,
                self.formatter.current_flow_tree,
                event.method_name,
                "running",
            )

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(source, event: MethodExecutionFinishedEvent):
            self.formatter.update_method_status(
                self.formatter.current_method_branch,
                self.formatter.current_flow_tree,
                event.method_name,
                "completed",
            )

        @crewai_event_bus.on(MethodExecutionFailedEvent)
        def on_method_execution_failed(source, event: MethodExecutionFailedEvent):
            self.formatter.update_method_status(
                self.formatter.current_method_branch,
                self.formatter.current_flow_tree,
                event.method_name,
                "failed",
            )

        # ----------- TOOL USAGE EVENTS -----------

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            if isinstance(source, LLM):
                self.formatter.handle_llm_tool_usage_started(
                    event.tool_name,
                    event.tool_args,
                )
            else:
                self.formatter.handle_tool_usage_started(
                    self.formatter.current_agent_branch,
                    event.tool_name,
                    self.formatter.current_crew_tree,
                )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event: ToolUsageFinishedEvent):
            if isinstance(source, LLM):
                self.formatter.handle_llm_tool_usage_finished(
                    event.tool_name,
                )
            else:
                self.formatter.handle_tool_usage_finished(
                    self.formatter.current_tool_branch,
                    event.tool_name,
                    self.formatter.current_crew_tree,
                )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolUsageErrorEvent):
            if isinstance(source, LLM):
                self.formatter.handle_llm_tool_usage_error(
                    event.tool_name,
                    event.error,
                )
            else:
                self.formatter.handle_tool_usage_error(
                    self.formatter.current_tool_branch,
                    event.tool_name,
                    event.error,
                    self.formatter.current_crew_tree,
                )

        # ----------- LLM EVENTS -----------

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(source, event: LLMCallStartedEvent):
            # Capture the returned tool branch and update the current_tool_branch reference
            thinking_branch = self.formatter.handle_llm_call_started(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )
            # Update the formatter's current_tool_branch to ensure proper cleanup
            if thinking_branch is not None:
                self.formatter.current_tool_branch = thinking_branch

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(source, event: LLMCallCompletedEvent):
            self.formatter.handle_llm_call_completed(
                self.formatter.current_tool_branch,
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_llm_call_failed(source, event: LLMCallFailedEvent):
            self.formatter.handle_llm_call_failed(
                self.formatter.current_tool_branch,
                event.error,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(source, event: LLMStreamChunkEvent):
            self.text_stream.write(event.chunk)

            self.text_stream.seek(self.next_chunk)

            # Read from the in-memory stream
            content = self.text_stream.read()
            print(content, end="", flush=True)
            self.next_chunk = self.text_stream.tell()

        # ----------- LLM GUARDRAIL EVENTS -----------

        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def on_llm_guardrail_started(source, event: LLMGuardrailStartedEvent):
            guardrail_str = str(event.guardrail)
            guardrail_name = (
                guardrail_str[:50] + "..." if len(guardrail_str) > 50 else guardrail_str
            )

            self.formatter.handle_guardrail_started(guardrail_name, event.retry_count)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def on_llm_guardrail_completed(source, event: LLMGuardrailCompletedEvent):
            self.formatter.handle_guardrail_completed(
                event.success, event.error, event.retry_count
            )

        @crewai_event_bus.on(CrewTestStartedEvent)
        def on_crew_test_started(source, event: CrewTestStartedEvent):
            cloned_crew = source.copy()
            self._telemetry.test_execution_span(
                cloned_crew,
                event.n_iterations,
                event.inputs,
                event.eval_llm or "",
            )

            self.formatter.handle_crew_test_started(
                event.crew_name or "Crew", source.id, event.n_iterations
            )

        @crewai_event_bus.on(CrewTestCompletedEvent)
        def on_crew_test_completed(source, event: CrewTestCompletedEvent):
            self.formatter.handle_crew_test_completed(
                self.formatter.current_flow_tree,
                event.crew_name or "Crew",
            )

        @crewai_event_bus.on(CrewTestFailedEvent)
        def on_crew_test_failed(source, event: CrewTestFailedEvent):
            self.formatter.handle_crew_test_failed(event.crew_name or "Crew")

        @crewai_event_bus.on(KnowledgeRetrievalStartedEvent)
        def on_knowledge_retrieval_started(
            source, event: KnowledgeRetrievalStartedEvent
        ):
            if self.knowledge_retrieval_in_progress:
                return

            self.knowledge_retrieval_in_progress = True

            self.formatter.handle_knowledge_retrieval_started(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(KnowledgeRetrievalCompletedEvent)
        def on_knowledge_retrieval_completed(
            source, event: KnowledgeRetrievalCompletedEvent
        ):
            if not self.knowledge_retrieval_in_progress:
                return

            self.knowledge_retrieval_in_progress = False
            self.formatter.handle_knowledge_retrieval_completed(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
                event.retrieved_knowledge,
            )

        @crewai_event_bus.on(KnowledgeQueryStartedEvent)
        def on_knowledge_query_started(source, event: KnowledgeQueryStartedEvent):
            pass

        @crewai_event_bus.on(KnowledgeQueryFailedEvent)
        def on_knowledge_query_failed(source, event: KnowledgeQueryFailedEvent):
            self.formatter.handle_knowledge_query_failed(
                self.formatter.current_agent_branch,
                event.error,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(KnowledgeQueryCompletedEvent)
        def on_knowledge_query_completed(source, event: KnowledgeQueryCompletedEvent):
            pass

        @crewai_event_bus.on(KnowledgeSearchQueryFailedEvent)
        def on_knowledge_search_query_failed(
            source, event: KnowledgeSearchQueryFailedEvent
        ):
            self.formatter.handle_knowledge_search_query_failed(
                self.formatter.current_agent_branch,
                event.error,
                self.formatter.current_crew_tree,
            )

        # ----------- REASONING EVENTS -----------

        @crewai_event_bus.on(AgentReasoningStartedEvent)
        def on_agent_reasoning_started(source, event: AgentReasoningStartedEvent):
            self.formatter.handle_reasoning_started(
                self.formatter.current_agent_branch,
                event.attempt,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(AgentReasoningCompletedEvent)
        def on_agent_reasoning_completed(source, event: AgentReasoningCompletedEvent):
            self.formatter.handle_reasoning_completed(
                event.plan,
                event.ready,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(AgentReasoningFailedEvent)
        def on_agent_reasoning_failed(source, event: AgentReasoningFailedEvent):
            self.formatter.handle_reasoning_failed(
                event.error,
                self.formatter.current_crew_tree,
            )

        # ----------- AGENT LOGGING EVENTS -----------

        @crewai_event_bus.on(AgentLogsStartedEvent)
        def on_agent_logs_started(source, event: AgentLogsStartedEvent):
            self.formatter.handle_agent_logs_started(
                event.agent_role,
                event.task_description,
                event.verbose,
            )

        @crewai_event_bus.on(AgentLogsExecutionEvent)
        def on_agent_logs_execution(source, event: AgentLogsExecutionEvent):
            self.formatter.handle_agent_logs_execution(
                event.agent_role,
                event.formatted_answer,
                event.verbose,
            )


event_listener = EventListener()
