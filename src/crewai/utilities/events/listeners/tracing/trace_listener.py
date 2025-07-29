import os
import uuid
from typing import Dict, List, Any, Optional

from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    AgentExecutionErrorEvent,
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

from .execution_context_tracker import ExecutionContextTracker, PrivacyFilter
from .trace_event_factory import TraceEventFactory
from .trace_batch_manager import TraceBatchManager
from .context_handlers import (
    ContextHandler,
    CrewContextHandler,
    TaskContextHandler,
    AgentContextHandler,
    FlowContextHandler,
    MethodContextHandler,
)
from .interfaces import ITraceSender, ConsoleTraceSender
from crewai.cli.authentication.token import get_auth_token
from crewai.cli.version import get_crewai_version


class TraceCollectionListener(BaseEventListener):
    """
    SOLID-compliant trace collection listener that orchestrates trace collection
    by delegating responsibilities to specialized components.

    Follows SOLID principles:
    - SRP: Only orchestrates, delegates specific tasks
    - OCP: Easy to add new context handlers without modification
    - LSP: All handlers are substitutable
    - ISP: Uses focused interfaces
    - DIP: Depends on abstractions (ITraceSender, ContextHandler)
    """

    trace_enabled: bool = False

    def __init__(
        self,
        context_tracker: Optional[ExecutionContextTracker] = None,
        privacy_filter: Optional[PrivacyFilter] = None,
        event_factory: Optional[TraceEventFactory] = None,
        batch_manager: Optional[TraceBatchManager] = None,
        trace_sender: Optional[ITraceSender] = None,
        context_handlers: Optional[List[ContextHandler]] = None,
    ):
        super().__init__()

        # Initialize dependencies with defaults (Dependency Injection)
        self.context_tracker = context_tracker or ExecutionContextTracker()
        self.privacy_filter = privacy_filter or PrivacyFilter(
            os.getenv("CREWAI_TRACING_PRIVACY_LEVEL", "full")
        )
        self.event_factory = event_factory or TraceEventFactory(
            self.context_tracker, self.privacy_filter
        )
        self.batch_manager = batch_manager or TraceBatchManager()
        self.trace_sender = trace_sender or ConsoleTraceSender()

        # Initialize context handlers
        if context_handlers:
            self.context_handlers = {h.__class__.__name__: h for h in context_handlers}
        else:
            # Default handlers for basic crew tracing
            self.context_handlers = {
                "CrewContextHandler": CrewContextHandler(),
                "TaskContextHandler": TaskContextHandler(),
                "AgentContextHandler": AgentContextHandler(),
                "FlowContextHandler": FlowContextHandler(),
                "MethodContextHandler": MethodContextHandler(),
            }

        self.trace_enabled = self._check_trace_enabled()

        if self.trace_enabled:
            print("🔍 Trace collection enabled")

    def _check_trace_enabled(self) -> bool:
        """Check if tracing should be enabled"""
        auth_token = get_auth_token()
        if not auth_token:
            return False
        return os.getenv("CREWAI_TRACING_ENABLED", "false").lower() == "true" or bool(
            os.getenv("CREWAI_USER_TOKEN")
        )

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
        if not self.trace_enabled:
            return

        # Register flow event handlers (NEW)
        self._register_flow_event_handlers(crewai_event_bus)

        # Register context start/end event handlers
        self._register_context_event_handlers(crewai_event_bus)

        # Register action event handlers (LLM, tools)
        self._register_action_event_handlers(crewai_event_bus)

    def _register_flow_event_handlers(self, event_bus):
        """Register handlers for flow events"""

        @event_bus.on(FlowCreatedEvent)
        def on_flow_created(source, event):
            # Don't initialize batch yet, wait for FlowStartedEvent
            pass  # Just log the creation for now

        @event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event):
            # Initialize batch on first flow event
            if not self.batch_manager.is_batch_initialized():
                self._initialize_flow_batch(source, event)
            self._handle_context_start("flow_started", source, event)

        @event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event):
            self._handle_context_end("flow_finished", source, event)
            # Send batch when flow completes
            if self.context_tracker.is_root_level():
                self._send_batch()

        @event_bus.on(MethodExecutionStartedEvent)
        def on_method_started(source, event):
            self._handle_context_start("method_execution_started", source, event)

        @event_bus.on(MethodExecutionFinishedEvent)
        def on_method_finished(source, event):
            self._handle_context_end("method_execution_finished", source, event)

        @event_bus.on(MethodExecutionFailedEvent)
        def on_method_failed(source, event):
            self._handle_context_end("method_execution_failed", source, event)

        @event_bus.on(FlowPlotEvent)
        def on_flow_plot(source, event):
            self._handle_action_event("flow_plot", source, event)

    def _register_context_event_handlers(self, event_bus):
        """Register handlers for context events (start/end)"""

        @event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            # Initialize batch on first crew event
            if not self.batch_manager.is_batch_initialized():
                self._initialize_batch(source, event)
            self._handle_context_start("crew_kickoff_started", source, event)

        @event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            self._handle_context_end("crew_kickoff_completed", source, event)
            # Send batch if at root level
            if self.context_tracker.is_root_level():
                self._send_batch()

        @event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event):
            self._handle_context_end("crew_kickoff_failed", source, event)
            if self.context_tracker.is_root_level():
                self._send_batch()

        @event_bus.on(TaskStartedEvent)
        def on_task_started(source, event):
            self._handle_context_start("task_started", source, event)

        @event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event):
            self._handle_context_end("task_completed", source, event)

        @event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event):
            self._handle_context_end("task_failed", source, event)

        @event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event):
            self._handle_context_start("agent_execution_started", source, event)

        @event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event):
            self._handle_context_end("agent_execution_completed", source, event)

        @event_bus.on(LiteAgentExecutionStartedEvent)
        def on_lite_agent_started(source, event):
            self._handle_context_start("lite_agent_execution_started", source, event)

        @event_bus.on(LiteAgentExecutionCompletedEvent)
        def on_lite_agent_completed(source, event):
            self._handle_context_end("lite_agent_execution_completed", source, event)

        @event_bus.on(LiteAgentExecutionErrorEvent)
        def on_lite_agent_error(source, event):
            self._handle_context_end("lite_agent_execution_error", source, event)

        @event_bus.on(AgentExecutionErrorEvent)
        def on_agent_error(source, event):
            self._handle_context_end("agent_execution_error", source, event)

    def _register_action_event_handlers(self, event_bus):
        """Register handlers for action events (LLM calls, tool usage)"""

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

        # @event_bus.on(AgentLogsStartedEvent)
        # def on_agent_logs_started(source, event):
        #     self._handle_action_event("agent_logs_started", source, event)

        # @event_bus.on(AgentLogsExecutionEvent)
        # def on_agent_logs_execution(source, event):
        #     self._handle_action_event("agent_logs_execution", source, event)

        # @event_bus.on(AgentEvaluationStartedEvent)
        # def on_agent_evaluation_started(source, event):
        #     self._handle_action_event("agent_evaluation_started", source, event)

        # @event_bus.on(AgentEvaluationCompletedEvent)
        # def on_agent_evaluation_completed(source, event):
        #     self._handle_action_event("agent_evaluation_completed", source, event)

        # @event_bus.on(AgentEvaluationFailedEvent)
        # def on_agent_evaluation_failed(source, event):
        #     self._handle_action_event("agent_evaluation_failed", source, event)

    def _initialize_batch(self, source: Any, event: Any):
        """Initialize trace batch"""
        user_context = self._get_user_context()
        execution_metadata = {
            "crew_name": getattr(event, "crew_name", "Unknown Crew"),
            "execution_start": event.timestamp if hasattr(event, "timestamp") else None,
            "crewai_version": get_crewai_version(),
            "privacy_level": self.privacy_filter.privacy_level,
        }

        self.batch_manager.initialize_batch(user_context, execution_metadata)

    def _initialize_flow_batch(self, source: Any, event: Any):
        """Initialize trace batch for Flow execution"""
        user_context = self._get_user_context()
        execution_metadata = {
            "flow_name": getattr(source, "__class__.__name__", "Unknown Flow"),
            "execution_start": event.timestamp if hasattr(event, "timestamp") else None,
            "crewai_version": get_crewai_version(),
            "privacy_level": self.privacy_filter.privacy_level,
            "execution_type": "flow",
        }

        self.batch_manager.initialize_batch(user_context, execution_metadata)

    def _handle_context_start(self, event_type: str, source: Any, event: Any):
        """Generic handler for context start events"""

        # Handle context tracking (delegates to appropriate handler)
        for handler in self.context_handlers.values():
            if handler.can_handle(event_type):
                handler.handle_start(source, event, self.context_tracker)
                break

        # Create and store trace event (delegates to factory)
        trace_id = self.batch_manager.get_trace_id() or "unknown"
        trace_event = self.event_factory.create_event(
            event_type, source, event, trace_id
        )
        self.batch_manager.add_event(trace_event)

        # Record timing for duration calculation
        timing_key = f"{event_type}_{getattr(source, 'id', 'unknown')}"
        self.batch_manager.record_start_time(timing_key)

    def _handle_context_end(self, event_type: str, source: Any, event: Any):
        """Generic handler for context end events"""

        # Handle context cleanup (delegates to appropriate handler)
        for handler in self.context_handlers.values():
            if handler.can_handle(event_type):
                handler.handle_end(source, event, self.context_tracker)
                break

        # Calculate duration
        timing_key = f"{event_type.replace('_completed', '_started').replace('_failed', '_started')}_{getattr(source, 'id', 'unknown')}"
        duration_ms = self.batch_manager.calculate_duration(timing_key)

        # Create trace event and add duration
        trace_id = self.batch_manager.get_trace_id() or "unknown"
        trace_event = self.event_factory.create_event(
            event_type, source, event, trace_id
        )
        if duration_ms > 0:
            trace_event.event_data["duration_ms"] = duration_ms

        self.batch_manager.add_event(trace_event)

    def _handle_action_event(self, event_type: str, source: Any, event: Any):
        """Generic handler for action events (LLM calls, tool usage)"""

        # Ensure batch is initialized (fallback)
        if not self.batch_manager.is_batch_initialized():
            user_context = self._get_user_context()
            execution_metadata = {
                "crew_name": getattr(source, "name", "Unknown Crew"),
                "crewai_version": get_crewai_version(),
                "privacy_level": self.privacy_filter.privacy_level,
            }
            self.batch_manager.initialize_batch(user_context, execution_metadata)

        # Create and store trace event
        trace_id = self.batch_manager.get_trace_id() or "unknown"
        trace_event = self.event_factory.create_event(
            event_type, source, event, trace_id
        )
        self.batch_manager.add_event(trace_event)

    def _send_batch(self):
        """Send finalized batch using the configured sender"""
        batch = self.batch_manager.finalize_batch()
        if batch:
            success = self.trace_sender.send_batch(batch)
            if not success:
                print("⚠️  Failed to send trace batch")
