from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, Any

from pydantic import Field, PrivateAttr

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.a2a_events import (
    A2AConversationCompletedEvent,
    A2AConversationStartedEvent,
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
    A2AMessageSentEvent,
    A2APollingStartedEvent,
    A2APollingStatusEvent,
    A2AResponseReceivedEvent,
)
from crewai.events.types.agent_events import (
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
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowPausedEvent,
    FlowStartedEvent,
    HumanFeedbackReceivedEvent,
    HumanFeedbackRequestedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionPausedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.knowledge_events import (
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
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
from crewai.events.types.mcp_events import (
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
)
from crewai.events.types.reasoning_events import (
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.llm import LLM
from crewai.task import Task
from crewai.telemetry.telemetry import Telemetry
from crewai.utilities import Logger
from crewai.utilities.constants import EMITTER_COLOR


if TYPE_CHECKING:
    from crewai.events.event_bus import CrewAIEventsBus


class EventListener(BaseEventListener):
    _instance: EventListener | None = None
    _initialized: bool = False
    _telemetry: Telemetry = PrivateAttr(default_factory=lambda: Telemetry())
    logger: Logger = Logger(verbose=True, default_color=EMITTER_COLOR)
    execution_spans: dict[Task, Any] = Field(default_factory=dict)
    next_chunk: int = 0
    text_stream: StringIO = StringIO()
    knowledge_retrieval_in_progress: bool = False
    knowledge_query_in_progress: bool = False

    def __new__(cls) -> EventListener:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            super().__init__()
            self._telemetry = Telemetry()
            self._telemetry.set_tracer()
            self.execution_spans = {}
            self._initialized = True
            self.formatter = ConsoleFormatter(verbose=True)

            # Initialize trace listener with formatter for memory event handling
            trace_listener = TraceCollectionListener()
            trace_listener.formatter = self.formatter

    # ----------- CREW EVENTS -----------

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source: Any, event: CrewKickoffStartedEvent) -> None:
            self.formatter.handle_crew_started(event.crew_name or "Crew", source.id)
            source._execution_span = self._telemetry.crew_execution_span(
                source, event.inputs
            )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source: Any, event: CrewKickoffCompletedEvent) -> None:
            # Handle telemetry
            final_string_output = event.output.raw
            self._telemetry.end_crew(source, final_string_output)

            self.formatter.handle_crew_status(
                event.crew_name or "Crew",
                source.id,
                "completed",
                final_string_output,
            )

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source: Any, event: CrewKickoffFailedEvent) -> None:
            self.formatter.handle_crew_status(
                event.crew_name or "Crew",
                source.id,
                "failed",
            )

        @crewai_event_bus.on(CrewTrainStartedEvent)
        def on_crew_train_started(_: Any, event: CrewTrainStartedEvent) -> None:
            self.formatter.handle_crew_train_started(
                event.crew_name or "Crew", str(event.timestamp)
            )

        @crewai_event_bus.on(CrewTrainCompletedEvent)
        def on_crew_train_completed(_: Any, event: CrewTrainCompletedEvent) -> None:
            self.formatter.handle_crew_train_completed(
                event.crew_name or "Crew", str(event.timestamp)
            )

        @crewai_event_bus.on(CrewTrainFailedEvent)
        def on_crew_train_failed(_: Any, event: CrewTrainFailedEvent) -> None:
            self.formatter.handle_crew_train_failed(event.crew_name or "Crew")

        @crewai_event_bus.on(CrewTestResultEvent)
        def on_crew_test_result(source: Any, event: CrewTestResultEvent) -> None:
            self._telemetry.individual_test_result_span(
                source.crew,
                event.quality,
                int(event.execution_duration),
                event.model,
            )

        # ----------- TASK EVENTS -----------

        def get_task_name(source: Any) -> str | None:
            return (
                source.name
                if hasattr(source, "name") and source.name
                else source.description
                if hasattr(source, "description") and source.description
                else None
            )

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source: Any, event: TaskStartedEvent) -> None:
            span = self._telemetry.task_started(crew=source.agent.crew, task=source)
            self.execution_spans[source] = span

            task_name = get_task_name(source)
            self.formatter.handle_task_started(source.id, task_name)

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source: Any, event: TaskCompletedEvent) -> None:
            # Handle telemetry
            span = self.execution_spans.pop(source, None)
            if span:
                self._telemetry.task_ended(span, source, source.agent.crew)

            # Pass task name if it exists
            task_name = get_task_name(source)
            self.formatter.handle_task_status(
                source.id, source.agent.role, "completed", task_name
            )

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source: Any, event: TaskFailedEvent) -> None:
            span = self.execution_spans.pop(source, None)
            if span:
                if source.agent and source.agent.crew:
                    self._telemetry.task_ended(span, source, source.agent.crew)

            # Pass task name if it exists
            task_name = get_task_name(source)
            self.formatter.handle_task_status(
                source.id, source.agent.role, "failed", task_name
            )

        # ----------- AGENT EVENTS -----------
        # ----------- LITE AGENT EVENTS -----------

        @crewai_event_bus.on(LiteAgentExecutionStartedEvent)
        def on_lite_agent_execution_started(
            _: Any, event: LiteAgentExecutionStartedEvent
        ) -> None:
            """Handle LiteAgent execution started event."""
            self.formatter.handle_lite_agent_execution(
                event.agent_info["role"], status="started", **event.agent_info
            )

        @crewai_event_bus.on(LiteAgentExecutionCompletedEvent)
        def on_lite_agent_execution_completed(
            _: Any, event: LiteAgentExecutionCompletedEvent
        ) -> None:
            """Handle LiteAgent execution completed event."""
            self.formatter.handle_lite_agent_execution(
                event.agent_info["role"], status="completed", **event.agent_info
            )

        @crewai_event_bus.on(LiteAgentExecutionErrorEvent)
        def on_lite_agent_execution_error(
            _: Any, event: LiteAgentExecutionErrorEvent
        ) -> None:
            """Handle LiteAgent execution error event."""
            self.formatter.handle_lite_agent_execution(
                event.agent_info["role"],
                status="failed",
                error=event.error,
                **event.agent_info,
            )

        # ----------- FLOW EVENTS -----------

        @crewai_event_bus.on(FlowCreatedEvent)
        def on_flow_created(_: Any, event: FlowCreatedEvent) -> None:
            self._telemetry.flow_creation_span(event.flow_name)

        @crewai_event_bus.on(FlowStartedEvent)
        def on_flow_started(source: Any, event: FlowStartedEvent) -> None:
            self._telemetry.flow_execution_span(
                event.flow_name, list(source._methods.keys())
            )
            self.formatter.handle_flow_created(event.flow_name, str(source.flow_id))
            self.formatter.handle_flow_started(event.flow_name, str(source.flow_id))

        @crewai_event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source: Any, event: FlowFinishedEvent) -> None:
            self.formatter.handle_flow_status(
                event.flow_name,
                source.flow_id,
            )

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(
            _: Any, event: MethodExecutionStartedEvent
        ) -> None:
            self.formatter.handle_method_status(
                event.method_name,
                "running",
            )

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(
            _: Any, event: MethodExecutionFinishedEvent
        ) -> None:
            self.formatter.handle_method_status(
                event.method_name,
                "completed",
            )

        @crewai_event_bus.on(MethodExecutionFailedEvent)
        def on_method_execution_failed(
            _: Any, event: MethodExecutionFailedEvent
        ) -> None:
            self.formatter.handle_method_status(
                event.method_name,
                "failed",
            )

        @crewai_event_bus.on(MethodExecutionPausedEvent)
        def on_method_execution_paused(
            _: Any, event: MethodExecutionPausedEvent
        ) -> None:
            self.formatter.handle_method_status(
                event.method_name,
                "paused",
            )

        @crewai_event_bus.on(FlowPausedEvent)
        def on_flow_paused(_: Any, event: FlowPausedEvent) -> None:
            self.formatter.handle_flow_status(
                event.flow_name,
                event.flow_id,
                "paused",
            )

        # ----------- HUMAN FEEDBACK EVENTS -----------
        @crewai_event_bus.on(HumanFeedbackRequestedEvent)
        def on_human_feedback_requested(
            _: Any, event: HumanFeedbackRequestedEvent
        ) -> None:
            """Handle human feedback requested event."""
            has_routing = event.emit is not None and len(event.emit) > 0
            self._telemetry.human_feedback_span(
                event_type="requested",
                has_routing=has_routing,
                num_outcomes=len(event.emit) if event.emit else 0,
            )

        @crewai_event_bus.on(HumanFeedbackReceivedEvent)
        def on_human_feedback_received(
            _: Any, event: HumanFeedbackReceivedEvent
        ) -> None:
            """Handle human feedback received event."""
            has_routing = event.outcome is not None
            self._telemetry.human_feedback_span(
                event_type="received",
                has_routing=has_routing,
                num_outcomes=0,
                feedback_provided=bool(event.feedback and event.feedback.strip()),
                outcome=event.outcome,
            )

        # ----------- TOOL USAGE EVENTS -----------
        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source: Any, event: ToolUsageStartedEvent) -> None:
            if isinstance(source, LLM):
                self.formatter.handle_llm_tool_usage_started(
                    event.tool_name,
                    event.tool_args,
                )
            else:
                self.formatter.handle_tool_usage_started(
                    event.tool_name,
                    event.tool_args,
                    event.run_attempts,
                )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source: Any, event: ToolUsageFinishedEvent) -> None:
            if isinstance(source, LLM):
                self.formatter.handle_llm_tool_usage_finished(
                    event.tool_name,
                )
            else:
                self.formatter.handle_tool_usage_finished(
                    event.tool_name,
                    event.output,
                    getattr(event, "run_attempts", None),
                )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source: Any, event: ToolUsageErrorEvent) -> None:
            if isinstance(source, LLM):
                self.formatter.handle_llm_tool_usage_error(
                    event.tool_name,
                    event.error,
                )
            else:
                self.formatter.handle_tool_usage_error(
                    event.tool_name,
                    event.error,
                    event.run_attempts,
                )

        # ----------- LLM EVENTS -----------

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(_: Any, event: LLMCallStartedEvent) -> None:
            self.text_stream = StringIO()
            self.next_chunk = 0

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(_: Any, event: LLMCallCompletedEvent) -> None:
            self.formatter.handle_llm_stream_completed()

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_llm_call_failed(_: Any, event: LLMCallFailedEvent) -> None:
            self.formatter.handle_llm_stream_completed()
            self.formatter.handle_llm_call_failed(event.error)

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(_: Any, event: LLMStreamChunkEvent) -> None:
            self.text_stream.write(event.chunk)
            self.text_stream.seek(self.next_chunk)
            self.text_stream.read()
            self.next_chunk = self.text_stream.tell()

            accumulated_text = self.text_stream.getvalue()
            self.formatter.handle_llm_stream_chunk(
                accumulated_text,
                event.call_type,
            )

        # ----------- LLM GUARDRAIL EVENTS -----------

        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def on_llm_guardrail_started(_: Any, event: LLMGuardrailStartedEvent) -> None:
            guardrail_str = str(event.guardrail)
            guardrail_name = (
                guardrail_str[:50] + "..." if len(guardrail_str) > 50 else guardrail_str
            )

            self.formatter.handle_guardrail_started(guardrail_name, event.retry_count)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def on_llm_guardrail_completed(
            _: Any, event: LLMGuardrailCompletedEvent
        ) -> None:
            self.formatter.handle_guardrail_completed(
                event.success, event.error, event.retry_count
            )

        @crewai_event_bus.on(CrewTestStartedEvent)
        def on_crew_test_started(source: Any, event: CrewTestStartedEvent) -> None:
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
        def on_crew_test_completed(_: Any, event: CrewTestCompletedEvent) -> None:
            self.formatter.handle_crew_test_completed(
                event.crew_name or "Crew",
            )

        @crewai_event_bus.on(CrewTestFailedEvent)
        def on_crew_test_failed(_: Any, event: CrewTestFailedEvent) -> None:
            self.formatter.handle_crew_test_failed(event.crew_name or "Crew")

        @crewai_event_bus.on(KnowledgeRetrievalStartedEvent)
        def on_knowledge_retrieval_started(
            _: Any, event: KnowledgeRetrievalStartedEvent
        ) -> None:
            if self.knowledge_retrieval_in_progress:
                return

            self.knowledge_retrieval_in_progress = True

            self.formatter.handle_knowledge_retrieval_started()

        @crewai_event_bus.on(KnowledgeRetrievalCompletedEvent)
        def on_knowledge_retrieval_completed(
            _: Any, event: KnowledgeRetrievalCompletedEvent
        ) -> None:
            if not self.knowledge_retrieval_in_progress:
                return

            self.knowledge_retrieval_in_progress = False
            self.formatter.handle_knowledge_retrieval_completed(
                event.retrieved_knowledge,
                event.query,
            )

        @crewai_event_bus.on(KnowledgeQueryFailedEvent)
        def on_knowledge_query_failed(_: Any, event: KnowledgeQueryFailedEvent) -> None:
            self.formatter.handle_knowledge_query_failed(event.error)

        @crewai_event_bus.on(KnowledgeQueryCompletedEvent)
        def on_knowledge_query_completed(
            _: Any, event: KnowledgeQueryCompletedEvent
        ) -> None:
            pass

        @crewai_event_bus.on(KnowledgeSearchQueryFailedEvent)
        def on_knowledge_search_query_failed(
            _: Any, event: KnowledgeSearchQueryFailedEvent
        ) -> None:
            self.formatter.handle_knowledge_search_query_failed(event.error)

        # ----------- REASONING EVENTS -----------

        @crewai_event_bus.on(AgentReasoningStartedEvent)
        def on_agent_reasoning_started(
            _: Any, event: AgentReasoningStartedEvent
        ) -> None:
            self.formatter.handle_reasoning_started(event.attempt)

        @crewai_event_bus.on(AgentReasoningCompletedEvent)
        def on_agent_reasoning_completed(
            _: Any, event: AgentReasoningCompletedEvent
        ) -> None:
            self.formatter.handle_reasoning_completed(
                event.plan,
                event.ready,
            )

        @crewai_event_bus.on(AgentReasoningFailedEvent)
        def on_agent_reasoning_failed(_: Any, event: AgentReasoningFailedEvent) -> None:
            self.formatter.handle_reasoning_failed(
                event.error,
            )

        # ----------- AGENT LOGGING EVENTS -----------

        @crewai_event_bus.on(AgentLogsStartedEvent)
        def on_agent_logs_started(_: Any, event: AgentLogsStartedEvent) -> None:
            self.formatter.handle_agent_logs_started(
                event.agent_role,
                event.task_description,
                event.verbose,
            )

        @crewai_event_bus.on(AgentLogsExecutionEvent)
        def on_agent_logs_execution(_: Any, event: AgentLogsExecutionEvent) -> None:
            self.formatter.handle_agent_logs_execution(
                event.agent_role,
                event.formatted_answer,
                event.verbose,
            )

        @crewai_event_bus.on(A2ADelegationStartedEvent)
        def on_a2a_delegation_started(_: Any, event: A2ADelegationStartedEvent) -> None:
            self.formatter.handle_a2a_delegation_started(
                event.endpoint,
                event.task_description,
                event.agent_id,
                event.is_multiturn,
                event.turn_number,
            )

        @crewai_event_bus.on(A2ADelegationCompletedEvent)
        def on_a2a_delegation_completed(
            _: Any, event: A2ADelegationCompletedEvent
        ) -> None:
            self.formatter.handle_a2a_delegation_completed(
                event.status,
                event.result,
                event.error,
                event.is_multiturn,
            )

        @crewai_event_bus.on(A2AConversationStartedEvent)
        def on_a2a_conversation_started(
            _: Any, event: A2AConversationStartedEvent
        ) -> None:
            # Store A2A agent name for display in conversation tree
            if event.a2a_agent_name:
                self.formatter._current_a2a_agent_name = event.a2a_agent_name

            self.formatter.handle_a2a_conversation_started(
                event.agent_id,
                event.endpoint,
            )

        @crewai_event_bus.on(A2AMessageSentEvent)
        def on_a2a_message_sent(_: Any, event: A2AMessageSentEvent) -> None:
            self.formatter.handle_a2a_message_sent(
                event.message,
                event.turn_number,
                event.agent_role,
            )

        @crewai_event_bus.on(A2AResponseReceivedEvent)
        def on_a2a_response_received(_: Any, event: A2AResponseReceivedEvent) -> None:
            self.formatter.handle_a2a_response_received(
                event.response,
                event.turn_number,
                event.status,
                event.agent_role,
            )

        @crewai_event_bus.on(A2AConversationCompletedEvent)
        def on_a2a_conversation_completed(
            _: Any, event: A2AConversationCompletedEvent
        ) -> None:
            self.formatter.handle_a2a_conversation_completed(
                event.status,
                event.final_result,
                event.error,
                event.total_turns,
            )

        @crewai_event_bus.on(A2APollingStartedEvent)
        def on_a2a_polling_started(_: Any, event: A2APollingStartedEvent) -> None:
            self.formatter.handle_a2a_polling_started(
                event.task_id,
                event.polling_interval,
                event.endpoint,
            )

        @crewai_event_bus.on(A2APollingStatusEvent)
        def on_a2a_polling_status(_: Any, event: A2APollingStatusEvent) -> None:
            self.formatter.handle_a2a_polling_status(
                event.task_id,
                event.state,
                event.elapsed_seconds,
                event.poll_count,
            )

        # ----------- MCP EVENTS -----------

        @crewai_event_bus.on(MCPConnectionStartedEvent)
        def on_mcp_connection_started(_: Any, event: MCPConnectionStartedEvent) -> None:
            self.formatter.handle_mcp_connection_started(
                event.server_name,
                event.server_url,
                event.transport_type,
                event.is_reconnect,
                event.connect_timeout,
            )

        @crewai_event_bus.on(MCPConnectionCompletedEvent)
        def on_mcp_connection_completed(
            _: Any, event: MCPConnectionCompletedEvent
        ) -> None:
            self.formatter.handle_mcp_connection_completed(
                event.server_name,
                event.server_url,
                event.transport_type,
                event.connection_duration_ms,
                event.is_reconnect,
            )

        @crewai_event_bus.on(MCPConnectionFailedEvent)
        def on_mcp_connection_failed(_: Any, event: MCPConnectionFailedEvent) -> None:
            self.formatter.handle_mcp_connection_failed(
                event.server_name,
                event.server_url,
                event.transport_type,
                event.error,
                event.error_type,
            )

        @crewai_event_bus.on(MCPToolExecutionStartedEvent)
        def on_mcp_tool_execution_started(
            _: Any, event: MCPToolExecutionStartedEvent
        ) -> None:
            self.formatter.handle_mcp_tool_execution_started(
                event.server_name,
                event.tool_name,
                event.tool_args,
            )

        @crewai_event_bus.on(MCPToolExecutionFailedEvent)
        def on_mcp_tool_execution_failed(
            _: Any, event: MCPToolExecutionFailedEvent
        ) -> None:
            self.formatter.handle_mcp_tool_execution_failed(
                event.server_name,
                event.tool_name,
                event.tool_args,
                event.error,
                event.error_type,
            )


event_listener = EventListener()
