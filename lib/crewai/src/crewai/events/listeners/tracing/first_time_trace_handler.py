from datetime import datetime, timezone
import logging
import uuid
import webbrowser

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

    def __init__(self) -> None:
        self.is_first_time = False
        self.collected_events = False
        self.trace_batch_id: str | None = None
        self.ephemeral_url: str | None = None
        self.batch_manager: TraceBatchManager | None = None

    def initialize_for_first_time_user(self) -> bool:
        """Check if this is first time and initialize collection."""
        self.is_first_time = should_auto_collect_first_time_traces()
        return self.is_first_time

    def set_batch_manager(self, batch_manager: TraceBatchManager) -> None:
        """Set reference to batch manager for sending events.

        Args:
            batch_manager: The trace batch manager instance.
        """
        self.batch_manager = batch_manager

    def mark_events_collected(self) -> None:
        """Mark that events have been collected during execution."""
        self.collected_events = True

    def handle_execution_completion(self) -> None:
        """Handle the completion flow as shown in your diagram."""
        if not self.is_first_time or not self.collected_events:
            return

        try:
            user_wants_traces = prompt_user_for_trace_viewing()

            if user_wants_traces:
                self._initialize_backend_and_send_events()

                if self.ephemeral_url:
                    self._display_ephemeral_trace_link()
            else:
                self._show_tracing_declined_message()

            mark_first_execution_completed(user_consented=user_wants_traces)

        except Exception as e:
            self._gracefully_fail(f"Error in trace handling: {e}")
            mark_first_execution_completed(user_consented=False)

    def _initialize_backend_and_send_events(self) -> None:
        """Initialize backend batch and send collected events."""
        if not self.batch_manager or not self.batch_manager.trace_batch_id:
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
                    skip_context_check=True,
                )

                if not self.batch_manager.trace_batch_id:
                    self._gracefully_fail(
                        "Backend batch creation failed, cannot send events."
                    )
                    self._reset_batch_state()
                    return

                self.batch_manager.backend_initialized = True

            # Capture values before send/finalize consume them
            events_count = len(self.batch_manager.event_buffer)
            batch_id = self.batch_manager.trace_batch_id
            # Read duration non-destructively — _finalize_backend_batch will consume it
            start_time = self.batch_manager.execution_start_times.get("execution")
            duration_ms = (
                int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                if start_time
                else 0
            )

            if self.batch_manager.event_buffer:
                send_status = self.batch_manager._send_events_to_backend()
                if send_status == 500 and self.batch_manager.trace_batch_id:
                    self.batch_manager._mark_batch_as_failed(
                        self.batch_manager.trace_batch_id,
                        "Error sending events to backend",
                    )
                    self._reset_batch_state()
                    return

            self.batch_manager._finalize_backend_batch(events_count)
            self.ephemeral_url = self.batch_manager.ephemeral_trace_url

            if not self.ephemeral_url:
                self._show_local_trace_message(events_count, duration_ms, batch_id)

            self._reset_batch_state()

        except Exception as e:
            self._gracefully_fail(f"Backend initialization failed: {e}")
            self._reset_batch_state()

    def _display_ephemeral_trace_link(self) -> None:
        """Display the ephemeral trace link to the user and automatically open browser."""
        console = Console()

        try:
            if self.ephemeral_url:
                webbrowser.open(self.ephemeral_url)
        except Exception:  # noqa: S110
            pass

        panel_content = f"""
🎉 Your First CrewAI Execution Trace is Ready!

View your execution details here:
{self.ephemeral_url}

This trace shows:
• Agent decisions and interactions
• Task execution timeline
• Tool usage and results
• LLM calls and responses

✅ Tracing has been enabled for future runs!
Your preference has been saved. Future Crew/Flow executions will automatically collect traces.

To disable tracing later, do any one of these:
• Set tracing=False in your Crew/Flow code
• Set CREWAI_TRACING_ENABLED=false in your project's .env file
• Run: crewai traces disable

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

    def _show_tracing_declined_message(self) -> None:
        """Show message when user declines tracing."""
        console = Console()

        panel_content = """
Info: Tracing has been disabled.

Your preference has been saved. Future Crew/Flow executions will not collect traces.

To enable tracing later, do any one of these:
• Set tracing=True in your Crew/Flow code
• Set CREWAI_TRACING_ENABLED=true in your project's .env file
• Run: crewai traces enable
        """.strip()

        panel = Panel(
            panel_content,
            title="Tracing Preference Saved",
            border_style="blue",
            padding=(1, 2),
        )

        console.print("\n")
        console.print(panel)
        console.print()

    def _reset_batch_state(self) -> None:
        """Reset batch manager state to allow future executions to re-initialize."""
        if not self.batch_manager:
            return
        self.batch_manager.batch_owner_type = None
        self.batch_manager.batch_owner_id = None
        self.batch_manager.current_batch = None
        self.batch_manager.event_buffer.clear()
        self.batch_manager.trace_batch_id = None
        self.batch_manager.is_current_batch_ephemeral = False
        self.batch_manager.backend_initialized = False
        self.batch_manager._cleanup_batch_data()

    def _gracefully_fail(self, error_message: str) -> None:
        """Handle errors gracefully without disrupting user experience."""
        console = Console()
        console.print(f"[yellow]Note: {error_message}[/yellow]")

        logger.debug(f"First-time trace error: {error_message}")

    def _show_local_trace_message(
        self, events_count: int = 0, duration_ms: int = 0, batch_id: str | None = None
    ) -> None:
        """Show message when traces were collected locally but couldn't be uploaded."""
        if self.batch_manager is None:
            return

        console = Console()

        panel_content = f"""
📊 Your execution traces were collected locally!

Unfortunately, we couldn't upload them to the server right now, but here's what we captured:
• {events_count} trace events
• Execution duration: {duration_ms}ms
• Batch ID: {batch_id}

✅ Tracing has been enabled for future runs!
Your preference has been saved. Future Crew/Flow executions will automatically collect traces.
The traces include agent decisions, task execution, and tool usage.

To disable tracing later, do any one of these:
• Set tracing=False in your Crew/Flow code
• Set CREWAI_TRACING_ENABLED=false in your project's .env file
• Run: crewai traces disable
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
