import logging
import uuid
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from crewai.events.listeners.tracing.trace_batch_manager import TraceBatchManager
from crewai.events.listeners.tracing.utils import (
    mark_first_execution_completed,
    prompt_user_for_trace_viewing,
    should_auto_collect_first_time_traces,
)

logger = logging.getLogger(__name__)


def _update_or_create_env_file():
    """Update or create .env file with CREWAI_TRACING_ENABLED=true."""
    env_path = Path(".env")
    env_content = ""
    variable_name = "CREWAI_TRACING_ENABLED"
    variable_value = "true"

    # Read existing content if file exists
    if env_path.exists():
        with open(env_path, "r") as f:
            env_content = f.read()

    # Check if CREWAI_TRACING_ENABLED is already set
    lines = env_content.splitlines()
    variable_exists = False
    updated_lines = []

    for line in lines:
        if line.strip().startswith(f"{variable_name}="):
            # Update existing variable
            updated_lines.append(f"{variable_name}={variable_value}")
            variable_exists = True
        else:
            updated_lines.append(line)

    # Add variable if it doesn't exist
    if not variable_exists:
        if updated_lines and not updated_lines[-1].strip():
            # If last line is empty, replace it
            updated_lines[-1] = f"{variable_name}={variable_value}"
        else:
            # Add new line and then the variable
            updated_lines.append(f"{variable_name}={variable_value}")

    # Write updated content
    with open(env_path, "w") as f:
        f.write("\n".join(updated_lines))
        if updated_lines:  # Add final newline if there's content
            f.write("\n")


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

                # Enable tracing for future runs by updating .env file
                try:
                    _update_or_create_env_file()
                except Exception:  # noqa: S110
                    pass

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
        """Display the ephemeral trace link to the user and automatically open browser."""
        console = Console()

        try:
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

✅ Tracing has been enabled for future runs! (CREWAI_TRACING_ENABLED=true added to .env)
You can also add tracing=True to your Crew(tracing=True) / Flow(tracing=True) for more control.

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

Tracing has been enabled for future runs! (CREWAI_TRACING_ENABLED=true added to .env)
The traces include agent decisions, task execution, and tool usage.
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
