from __future__ import annotations

from io import StringIO
import threading
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
    A2AResponseReceivedEvent,
)
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
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
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
from crewai.events.types.mcp_events import (
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionCompletedEvent,
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
    method_branches: dict[str, Any] = Field(default_factory=dict)

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
            self.method_branches = {}
            self._initialized = True
            self.formatter = ConsoleFormatter(verbose=True)
            self._crew_tree_lock = threading.Condition()

            # Initialize trace listener with formatter for memory event handling
            trace_listener = TraceCollectionListener()
            trace_listener.formatter = self.formatter

    # ----------- CREW EVENTS -----------

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source: Any, event: CrewKickoffStartedEvent) -> None:
            with self._crew_tree_lock:
                self.formatter.create_crew_tree(event.crew_name or "Crew", source.id)
                source._execution_span = self._telemetry.crew_execution_span(
                    source, event.inputs
                )
                self._crew_tree_lock.notify_all()

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source: Any, event: CrewKickoffCompletedEvent) -> None:
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
        def on_crew_failed(source: Any, event: CrewKickoffFailedEvent) -> None:
            self.formatter.update_crew_tree(
                self.formatter.current_crew_tree,
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

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source: Any, event: TaskStartedEvent) -> None:
            span = self._telemetry.task_started(crew=source.agent.crew, task=source)
            self.execution_spans[source] = span

            with self._crew_tree_lock:
                self._crew_tree_lock.wait_for(
                    lambda: self.formatter.current_crew_tree is not None, timeout=5.0
                )

            if self.formatter.current_crew_tree is not None:
                task_name = (
                    source.name if hasattr(source, "name") and source.name else None
                )
                self.formatter.create_task_branch(
                    self.formatter.current_crew_tree, source.id, task_name
                )

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source: Any, event: TaskCompletedEvent) -> None:
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
        def on_task_failed(source: Any, event: TaskFailedEvent) -> None:
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
        def on_agent_execution_started(
            _: Any, event: AgentExecutionStartedEvent
        ) -> None:
            self.formatter.create_agent_branch(
                self.formatter.current_task_branch,
                event.agent.role,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(
            _: Any, event: AgentExecutionCompletedEvent
        ) -> None:
            self.formatter.update_agent_status(
                self.formatter.current_agent_branch,
                event.agent.role,
                self.formatter.current_crew_tree,
            )

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
            tree = self.formatter.create_flow_tree(event.flow_name, str(source.flow_id))
            self.formatter.current_flow_tree = tree
            self.formatter.start_flow(event.flow_name, str(source.flow_id))

        @crewai_event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source: Any, event: FlowFinishedEvent) -> None:
            self.formatter.update_flow_status(
                self.formatter.current_flow_tree, event.flow_name, source.flow_id
            )

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(
            _: Any, event: MethodExecutionStartedEvent
        ) -> None:
            method_branch = self.method_branches.get(event.method_name)
            updated_branch = self.formatter.update_method_status(
                method_branch,
                self.formatter.current_flow_tree,
                event.method_name,
                "running",
            )
            self.method_branches[event.method_name] = updated_branch

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(
            _: Any, event: MethodExecutionFinishedEvent
        ) -> None:
            method_branch = self.method_branches.get(event.method_name)
            updated_branch = self.formatter.update_method_status(
                method_branch,
                self.formatter.current_flow_tree,
                event.method_name,
                "completed",
            )
            self.method_branches[event.method_name] = updated_branch

        @crewai_event_bus.on(MethodExecutionFailedEvent)
        def on_method_execution_failed(
            _: Any, event: MethodExecutionFailedEvent
        ) -> None:
            method_branch = self.method_branches.get(event.method_name)
            updated_branch = self.formatter.update_method_status(
                method_branch,
                self.formatter.current_flow_tree,
                event.method_name,
                "failed",
            )
            self.method_branches[event.method_name] = updated_branch

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
                    self.formatter.current_agent_branch,
                    event.tool_name,
                    self.formatter.current_crew_tree,
                )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source: Any, event: ToolUsageFinishedEvent) -> None:
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
        def on_tool_usage_error(source: Any, event: ToolUsageErrorEvent) -> None:
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
        def on_llm_call_started(_: Any, event: LLMCallStartedEvent) -> None:
            self.text_stream = StringIO()
            self.next_chunk = 0
            # Capture the returned tool branch and update the current_tool_branch reference
            thinking_branch = self.formatter.handle_llm_call_started(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )
            # Update the formatter's current_tool_branch to ensure proper cleanup
            if thinking_branch is not None:
                self.formatter.current_tool_branch = thinking_branch

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(_: Any, event: LLMCallCompletedEvent) -> None:
            self.formatter.handle_llm_stream_completed()
            self.formatter.handle_llm_call_completed(
                self.formatter.current_tool_branch,
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_llm_call_failed(_: Any, event: LLMCallFailedEvent) -> None:
            self.formatter.handle_llm_stream_completed()
            self.formatter.handle_llm_call_failed(
                self.formatter.current_tool_branch,
                event.error,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(_: Any, event: LLMStreamChunkEvent) -> None:
            self.text_stream.write(event.chunk)
            self.text_stream.seek(self.next_chunk)
            self.text_stream.read()
            self.next_chunk = self.text_stream.tell()

            accumulated_text = self.text_stream.getvalue()
            self.formatter.handle_llm_stream_chunk(
                event.chunk,
                accumulated_text,
                self.formatter.current_crew_tree,
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
                self.formatter.current_flow_tree,
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

            self.formatter.handle_knowledge_retrieval_started(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(KnowledgeRetrievalCompletedEvent)
        def on_knowledge_retrieval_completed(
            _: Any, event: KnowledgeRetrievalCompletedEvent
        ) -> None:
            if not self.knowledge_retrieval_in_progress:
                return

            self.knowledge_retrieval_in_progress = False
            self.formatter.handle_knowledge_retrieval_completed(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
                event.retrieved_knowledge,
            )

        @crewai_event_bus.on(KnowledgeQueryStartedEvent)
        def on_knowledge_query_started(
            _: Any, event: KnowledgeQueryStartedEvent
        ) -> None:
            pass

        @crewai_event_bus.on(KnowledgeQueryFailedEvent)
        def on_knowledge_query_failed(_: Any, event: KnowledgeQueryFailedEvent) -> None:
            self.formatter.handle_knowledge_query_failed(
                self.formatter.current_agent_branch,
                event.error,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(KnowledgeQueryCompletedEvent)
        def on_knowledge_query_completed(
            _: Any, event: KnowledgeQueryCompletedEvent
        ) -> None:
            pass

        @crewai_event_bus.on(KnowledgeSearchQueryFailedEvent)
        def on_knowledge_search_query_failed(
            _: Any, event: KnowledgeSearchQueryFailedEvent
        ) -> None:
            self.formatter.handle_knowledge_search_query_failed(
                self.formatter.current_agent_branch,
                event.error,
                self.formatter.current_crew_tree,
            )

        # ----------- REASONING EVENTS -----------

        @crewai_event_bus.on(AgentReasoningStartedEvent)
        def on_agent_reasoning_started(
            _: Any, event: AgentReasoningStartedEvent
        ) -> None:
            self.formatter.handle_reasoning_started(
                self.formatter.current_agent_branch,
                event.attempt,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(AgentReasoningCompletedEvent)
        def on_agent_reasoning_completed(
            _: Any, event: AgentReasoningCompletedEvent
        ) -> None:
            self.formatter.handle_reasoning_completed(
                event.plan,
                event.ready,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(AgentReasoningFailedEvent)
        def on_agent_reasoning_failed(_: Any, event: AgentReasoningFailedEvent) -> None:
            self.formatter.handle_reasoning_failed(
                event.error,
                self.formatter.current_crew_tree,
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

        @crewai_event_bus.on(MCPToolExecutionCompletedEvent)
        def on_mcp_tool_execution_completed(
            _: Any, event: MCPToolExecutionCompletedEvent
        ) -> None:
            self.formatter.handle_mcp_tool_execution_completed(
                event.server_name,
                event.tool_name,
                event.tool_args,
                event.result,
                event.execution_duration_ms,
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
