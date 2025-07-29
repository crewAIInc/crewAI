from abc import ABC, abstractmethod
from typing import Any

from .execution_context_tracker import ExecutionContextTracker


class ContextHandler(ABC):
    """Abstract base for context handlers (OCP compliance)"""

    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the given event type"""
        pass

    @abstractmethod
    def handle_start(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Handle context start event"""
        pass

    @abstractmethod
    def handle_end(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Handle context end event"""
        pass


class CrewContextHandler(ContextHandler):
    """Handle crew-level context events"""

    def can_handle(self, event_type: str) -> bool:
        return event_type.startswith("crew_kickoff_")

    def handle_start(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Push crew context onto stack"""
        crew_id = str(getattr(source, "id", "unknown"))
        crew_name = getattr(event, "crew_name", None) or "Unknown Crew"
        tracker.push_context("crew", crew_id, crew_name)

    def handle_end(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Pop crew context from stack"""
        tracker.pop_context("crew")


class TaskContextHandler(ContextHandler):
    """Handle task-level context events"""

    def can_handle(self, event_type: str) -> bool:
        return event_type.startswith("task_")

    def handle_start(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Push task context onto stack"""
        task_id = str(getattr(source, "id", "unknown"))
        task_name = getattr(source, "name", None)
        tracker.push_context("task", task_id, task_name)

    def handle_end(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Pop task context from stack"""
        tracker.pop_context("task")


class AgentContextHandler(ContextHandler):
    """Handle agent-level context events"""

    def can_handle(self, event_type: str) -> bool:
        return event_type.startswith("agent_execution_")

    def handle_start(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Push agent context onto stack"""
        agent_id = str(getattr(source, "id", "unknown"))
        agent_role = (
            getattr(event.agent, "role", None) if hasattr(event, "agent") else None
        )
        tracker.push_context("agent", agent_id, agent_role)

    def handle_end(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Pop agent context from stack"""
        tracker.pop_context("agent")


class FlowContextHandler(ContextHandler):
    """Handle flow-level context events"""

    def can_handle(self, event_type: str) -> bool:
        return event_type.startswith("flow_")

    def handle_start(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Push flow context onto stack"""
        flow_id = getattr(source, "flow_id", "unknown")
        flow_name = getattr(event, "flow_name", None)
        tracker.push_context("flow", flow_id, flow_name)

    def handle_end(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Pop flow context from stack"""
        tracker.pop_context("flow")


class MethodContextHandler(ContextHandler):
    """Handle method-level context events"""

    def can_handle(self, event_type: str) -> bool:
        return event_type.startswith("method_execution_")

    def handle_start(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Push method context onto stack"""
        flow_name = getattr(event, "flow_name", "unknown")
        method_name = getattr(event, "method_name", "unknown")
        method_id = f"{flow_name}::{method_name}"
        tracker.push_context("method", method_id, method_name)

    def handle_end(self, source: Any, event: Any, tracker: ExecutionContextTracker):
        """Pop method context from stack"""
        tracker.pop_context("method")
