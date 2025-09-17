import logging
import uuid

from rich.console import Console
from rich.panel import Panel

from crewai.events.listeners.tracing.trace_batch_manager import TraceBatchManager
from crewai.events.listeners.tracing.utils import (
    mark_first_execution_completed,
    prompt_user_for_trace_viewing,
    should_auto_collect_first_time_traces,
)

logger = logging.getLogger(__name__)


class FirstTimeTraceHandler:
    """Handles the first-time user trace collection and display flow."""

    def __init__(self):
        self.is_first_time: bool = False
        self.collected_events: bool = False
        self.trace_batch_id: str | None = None
        self.ephemeral_url: str | None = None
        self.batch_manager: TraceBatchManager | None = None

    def initialize_for_first_time_user(self) -> bool:
        """Check if this is first time and initialize collection."""
        self.is_first_time = should_auto_collect_first_time_traces()
        return self.is_first_time

    def set_batch_manager(self, batch_manager: TraceBatchManager):
        """Set reference to batch manager for sending events."""
        self.batch_manager = batch_manager

    def mark_events_collected(self):
        """Mark that events have been collected during execution."""
        self.collected_events = True

    def handle_execution_completion(self):
        """Handle the completion flow as shown in your diagram."""
        if not self.is_first_time or not self.collected_events:
            return

        try:
            user_wants_traces = prompt_user_for_trace_viewing(timeout_seconds=20)

            if user_wants_traces:
                self._initialize_backend_and_send_events()

                if self.ephemeral_url:
                    self._display_ephemeral_trace_link()

            mark_first_execution_completed()

        except Exception as e:
            self._gracefully_fail(f"Error in trace handling: {e}")
            mark_first_execution_completed()

    def _initialize_backend_and_send_events(self):
        """Initialize backend batch and send collected events."""
        if not self.batch_manager:
            return

        try:
            if not self.batch_manager.backend_initialized:
                original_metadata = (
                    self.batch_manager.current_batch.execution_metadata
                    if self.batch_manager.current_batch
                    else {}
                )

                user_context = {
                    "privacy_level": "standard",
                    "user_id": "first_time_user",
                    "session_id": str(uuid.uuid4()),
                    "trace_id": self.batch_manager.trace_batch_id,
                }

                execution_metadata = {
                    "execution_type": original_metadata.get("execution_type", "crew"),
                    "crew_name": original_metadata.get(
                        "crew_name", "First Time Execution"
                    ),
                    "flow_name": original_metadata.get("flow_name"),
                    "agent_count": original_metadata.get("agent_count", 1),
                    "task_count": original_metadata.get("task_count", 1),
                    "crewai_version": original_metadata.get("crewai_version"),
                }

                self.batch_manager._initialize_backend_batch(
                    user_context=user_context,
                    execution_metadata=execution_metadata,
                    use_ephemeral=True,
                )
                self.batch_manager.backend_initialized = True

            if self.batch_manager.event_buffer:
                self.batch_manager._send_events_to_backend()

            self.batch_manager.finalize_batch()
            self.ephemeral_url = self.batch_manager.ephemeral_trace_url

            if not self.ephemeral_url:
                self._show_local_trace_message()

        except Exception as e:
            self._gracefully_fail(f"Backend initialization failed: {e}")

    def _display_ephemeral_trace_link(self):
        """Display the ephemeral trace link to the user."""
        console = Console()

        panel_content = f"""
🎉 Your First CrewAI Execution Trace is Ready!

View your execution details here:
{self.ephemeral_url}

This trace shows:
• Agent decisions and interactions
• Task execution timeline
• Tool usage and results
• LLM calls and responses

To use traces add tracing=True to your Crew(tracing=True) / Flow(tracing=True)

📝 Note: This link will expire in 24 hours.
        """.strip()

        panel = Panel(
            panel_content,
            title="🔍 Execution Trace Generated",
            border_style="bright_green",
            padding=(1, 2),
        )

        console.print("\n")
        console.print(panel)
        console.print()

    def _gracefully_fail(self, error_message: str):
        """Handle errors gracefully without disrupting user experience."""
        console = Console()
        console.print(f"[yellow]Note: {error_message}[/yellow]")

        logger.debug(f"First-time trace error: {error_message}")

    def _show_local_trace_message(self):
        """Show message when traces were collected locally but couldn't be uploaded."""
        console = Console()

        panel_content = f"""
📊 Your execution traces were collected locally!

Unfortunately, we couldn't upload them to the server right now, but here's what we captured:
• {len(self.batch_manager.event_buffer)} trace events
• Execution duration: {self.batch_manager.calculate_duration("execution")}ms
• Batch ID: {self.batch_manager.trace_batch_id}

The traces include agent decisions, task execution, and tool usage.
Try running with CREWAI_TRACING_ENABLED=true next time for persistent traces.
    """.strip()

        panel = Panel(
            panel_content,
            title="🔍 Local Traces Collected",
            border_style="yellow",
            padding=(1, 2),
        )

        console.print("\n")
        console.print(panel)
        console.print()
