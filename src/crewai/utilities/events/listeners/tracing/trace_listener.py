import os
import uuid

from typing import Dict, Any, Optional

from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    AgentExecutionErrorEvent,
)
from crewai.utilities.events.listeners.tracing.types import TraceEvent
from crewai.utilities.events.reasoning_events import (
    AgentReasoningStartedEvent,
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
)
from crewai.utilities.events.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.utilities.events.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.utilities.events.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)

from crewai.utilities.events.flow_events import (
    FlowCreatedEvent,
    FlowStartedEvent,
    FlowFinishedEvent,
    MethodExecutionStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionFailedEvent,
    FlowPlotEvent,
)
from crewai.utilities.events.llm_guardrail_events import (
    LLMGuardrailStartedEvent,
    LLMGuardrailCompletedEvent,
)
from crewai.utilities.serialization import to_serializable


from .trace_batch_manager import TraceBatchManager

from crewai.utilities.events.memory_events import (
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
)

from crewai.cli.authentication.token import AuthError, get_auth_token
from crewai.cli.version import get_crewai_version


class TraceCollectionListener(BaseEventListener):
    """
    Trace collection listener that orchestrates trace collection
    """

    complex_events = ["task_started", "llm_call_started", "llm_call_completed"]

    _instance = None
    _initialized = False

    def __new__(cls, batch_manager=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        batch_manager: Optional[TraceBatchManager] = None,
    ):
        if self._initialized:
            return

        super().__init__()
        self.batch_manager = batch_manager or TraceBatchManager()
        self._initialized = True

    def _check_authenticated(self) -> bool:
        """Check if tracing should be enabled"""
        try:
            res = bool(get_auth_token())
            return res
        except AuthError:
            return False

    def _get_user_context(self) -> Dict[str, str]:
        """Extract user context for tracing"""
        return {
            "user_id": os.getenv("CREWAI_USER_ID", "anonymous"),
            "organization_id": os.getenv("CREWAI_ORG_ID", ""),
            "session_id": str(uuid.uuid4()),
            "trace_id": str(uuid.uuid4()),
        }

    def setup_listeners(self, crewai_event_bus):
        """Setup event listeners - delegates to specific handlers"""

        self._register_flow_event_handlers(crewai_event_bus)
        self._register_context_event_handlers(crewai_event_bus)
        self._register_action_event_handlers(crewai_event_bus)

    def _register_flow_event_handlers(self, event_bus):
        """Register handlers for flow events"""

        @event_bus.on(FlowCreatedEvent)
        def on_flow_created(source, event):
            pass

        @event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event):
            if not self.batch_manager.is_batch_initialized():
                self._initialize_flow_batch(source, event)
            self._handle_trace_event("flow_started", source, event)

        @event_bus.on(MethodExecutionStartedEvent)
        def on_method_started(source, event):
            self._handle_trace_event("method_execution_started", source, event)

        @event_bus.on(MethodExecutionFinishedEvent)
        def on_method_finished(source, event):
            self._handle_trace_event("method_execution_finished", source, event)

        @event_bus.on(MethodExecutionFailedEvent)
        def on_method_failed(source, event):
            self._handle_trace_event("method_execution_failed", source, event)

        @event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event):
            self._handle_trace_event("flow_finished", source, event)
            self.batch_manager.finalize_batch()

        @event_bus.on(FlowPlotEvent)
        def on_flow_plot(source, event):
            self._handle_action_event("flow_plot", source, event)

    def _register_context_event_handlers(self, event_bus):
        """Register handlers for context events (start/end)"""

        @event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            if not self.batch_manager.is_batch_initialized():
                self._initialize_crew_batch(source, event)
            self._handle_trace_event("crew_kickoff_started", source, event)

        @event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            self._handle_trace_event("crew_kickoff_completed", source, event)
            self.batch_manager.finalize_batch()

        @event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event):
            self._handle_trace_event("crew_kickoff_failed", source, event)
            self.batch_manager.finalize_batch()

        @event_bus.on(TaskStartedEvent)
        def on_task_started(source, event):
            self._handle_trace_event("task_started", source, event)

        @event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event):
            self._handle_trace_event("task_completed", source, event)

        @event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event):
            self._handle_trace_event("task_failed", source, event)

        @event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event):
            self._handle_trace_event("agent_execution_started", source, event)

        @event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event):
            self._handle_trace_event("agent_execution_completed", source, event)

        @event_bus.on(LiteAgentExecutionStartedEvent)
        def on_lite_agent_started(source, event):
            self._handle_trace_event("lite_agent_execution_started", source, event)

        @event_bus.on(LiteAgentExecutionCompletedEvent)
        def on_lite_agent_completed(source, event):
            self._handle_trace_event("lite_agent_execution_completed", source, event)

        @event_bus.on(LiteAgentExecutionErrorEvent)
        def on_lite_agent_error(source, event):
            self._handle_trace_event("lite_agent_execution_error", source, event)

        @event_bus.on(AgentExecutionErrorEvent)
        def on_agent_error(source, event):
            self._handle_trace_event("agent_execution_error", source, event)

        @event_bus.on(LLMGuardrailStartedEvent)
        def on_guardrail_started(source, event):
            self._handle_trace_event("llm_guardrail_started", source, event)

        @event_bus.on(LLMGuardrailCompletedEvent)
        def on_guardrail_completed(source, event):
            self._handle_trace_event("llm_guardrail_completed", source, event)

    def _register_action_event_handlers(self, event_bus):
        """Register handlers for action events (LLM calls, tool usage, memory)"""

        @event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(source, event):
            self._handle_action_event("llm_call_started", source, event)

        @event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(source, event):
            self._handle_action_event("llm_call_completed", source, event)

        @event_bus.on(LLMCallFailedEvent)
        def on_llm_call_failed(source, event):
            self._handle_action_event("llm_call_failed", source, event)

        @event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source, event):
            self._handle_action_event("tool_usage_started", source, event)

        @event_bus.on(ToolUsageFinishedEvent)
        def on_tool_finished(source, event):
            self._handle_action_event("tool_usage_finished", source, event)

        @event_bus.on(ToolUsageErrorEvent)
        def on_tool_error(source, event):
            self._handle_action_event("tool_usage_error", source, event)

        @event_bus.on(MemoryQueryStartedEvent)
        def on_memory_query_started(source, event):
            self._handle_action_event("memory_query_started", source, event)

        @event_bus.on(MemoryQueryCompletedEvent)
        def on_memory_query_completed(source, event):
            self._handle_action_event("memory_query_completed", source, event)

        @event_bus.on(MemoryQueryFailedEvent)
        def on_memory_query_failed(source, event):
            self._handle_action_event("memory_query_failed", source, event)

        @event_bus.on(MemorySaveStartedEvent)
        def on_memory_save_started(source, event):
            self._handle_action_event("memory_save_started", source, event)

        @event_bus.on(MemorySaveCompletedEvent)
        def on_memory_save_completed(source, event):
            self._handle_action_event("memory_save_completed", source, event)

        @event_bus.on(MemorySaveFailedEvent)
        def on_memory_save_failed(source, event):
            self._handle_action_event("memory_save_failed", source, event)

        @event_bus.on(AgentReasoningStartedEvent)
        def on_agent_reasoning_started(source, event):
            self._handle_action_event("agent_reasoning_started", source, event)

        @event_bus.on(AgentReasoningCompletedEvent)
        def on_agent_reasoning_completed(source, event):
            self._handle_action_event("agent_reasoning_completed", source, event)

        @event_bus.on(AgentReasoningFailedEvent)
        def on_agent_reasoning_failed(source, event):
            self._handle_action_event("agent_reasoning_failed", source, event)

    def _initialize_crew_batch(self, source: Any, event: Any):
        """Initialize trace batch"""
        user_context = self._get_user_context()
        execution_metadata = {
            "crew_name": getattr(event, "crew_name", "Unknown Crew"),
            "execution_start": event.timestamp if hasattr(event, "timestamp") else None,
            "crewai_version": get_crewai_version(),
        }

        self._initialize_batch(user_context, execution_metadata)

    def _initialize_flow_batch(self, source: Any, event: Any):
        """Initialize trace batch for Flow execution"""
        user_context = self._get_user_context()
        execution_metadata = {
            "flow_name": getattr(event, "flow_name", "Unknown Flow"),
            "execution_start": event.timestamp if hasattr(event, "timestamp") else None,
            "crewai_version": get_crewai_version(),
            "execution_type": "flow",
        }

        self._initialize_batch(user_context, execution_metadata)

    def _initialize_batch(
        self, user_context: Dict[str, str], execution_metadata: Dict[str, Any]
    ):
        """Initialize trace batch if ephemeral"""
        if not self._check_authenticated():
            self.batch_manager.initialize_batch(
                user_context, execution_metadata, use_ephemeral=True
            )
        else:
            self.batch_manager.initialize_batch(
                user_context, execution_metadata, use_ephemeral=False
            )

    def _handle_trace_event(self, event_type: str, source: Any, event: Any):
        """Generic handler for context end events"""

        trace_event = self._create_trace_event(event_type, source, event)

        self.batch_manager.add_event(trace_event)

    def _handle_action_event(self, event_type: str, source: Any, event: Any):
        """Generic handler for action events (LLM calls, tool usage)"""

        if not self.batch_manager.is_batch_initialized():
            user_context = self._get_user_context()
            execution_metadata = {
                "crew_name": getattr(source, "name", "Unknown Crew"),
                "crewai_version": get_crewai_version(),
            }
            self.batch_manager.initialize_batch(user_context, execution_metadata)

        trace_event = self._create_trace_event(event_type, source, event)
        self.batch_manager.add_event(trace_event)

    def _create_trace_event(
        self, event_type: str, source: Any, event: Any
    ) -> TraceEvent:
        """Create a trace event"""
        trace_event = TraceEvent(
            type=event_type,
        )

        trace_event.event_data = self._build_event_data(event_type, event, source)
        return trace_event

    def _build_event_data(
        self, event_type: str, event: Any, source: Any
    ) -> Dict[str, Any]:
        """Build event data"""
        if event_type not in self.complex_events:
            return self._safe_serialize_to_dict(event)
        elif event_type == "task_started":
            return {
                "task_description": event.task.description,
                "expected_output": event.task.expected_output,
                "task_name": event.task.name,
                "context": event.context,
                "agent": source.agent.role,
            }
        elif event_type == "llm_call_started":
            return self._safe_serialize_to_dict(event)
        elif event_type == "llm_call_completed":
            return self._safe_serialize_to_dict(event)
        else:
            return {
                "event_type": event_type,
                "event": self._safe_serialize_to_dict(event),
                "source": source,
            }

    # TODO: move to utils
    def _safe_serialize_to_dict(
        self, obj, exclude: set[str] | None = None
    ) -> Dict[str, Any]:
        """Safely serialize an object to a dictionary for event data."""
        try:
            serialized = to_serializable(obj, exclude)
            if isinstance(serialized, dict):
                return serialized
            else:
                return {"serialized_data": serialized}
        except Exception as e:
            return {"serialization_error": str(e), "object_type": type(obj).__name__}

    # TODO: move to utils
    def _truncate_messages(self, messages, max_content_length=500, max_messages=5):
        """Truncate message content and limit number of messages"""
        if not messages or not isinstance(messages, list):
            return messages

        # Limit number of messages
        limited_messages = messages[:max_messages]

        # Truncate each message content
        for msg in limited_messages:
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                if len(content) > max_content_length:
                    msg["content"] = content[:max_content_length] + "..."

        return limited_messages
