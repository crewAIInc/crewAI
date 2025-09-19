import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import getLogger
from typing import Any

from rich.console import Console
from rich.panel import Panel

from crewai.cli.authentication.token import AuthError, get_auth_token
from crewai.cli.plus_api import PlusAPI
from crewai.cli.version import get_crewai_version
from crewai.events.listeners.tracing.types import TraceEvent
from crewai.events.listeners.tracing.utils import should_auto_collect_first_time_traces
from crewai.utilities.constants import CREWAI_BASE_URL

logger = getLogger(__name__)


@dataclass
class TraceBatch:
    """Batch of events to send to backend"""

    version: str = field(default_factory=get_crewai_version)
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_context: dict[str, str] = field(default_factory=dict)
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "batch_id": self.batch_id,
            "user_context": self.user_context,
            "execution_metadata": self.execution_metadata,
            "events": [event.to_dict() for event in self.events],
        }


class TraceBatchManager:
    """Single responsibility: Manage batches and event buffering"""

    def __init__(self):
        self.is_current_batch_ephemeral: bool = False
        self.trace_batch_id: str | None = None
        self.current_batch: TraceBatch | None = None
        self.event_buffer: list[TraceEvent] = []
        self.execution_start_times: dict[str, datetime] = {}
        self.batch_owner_type: str | None = None
        self.batch_owner_id: str | None = None
        self.backend_initialized: bool = False
        self.ephemeral_trace_url: str | None = None
        try:
            self.plus_api = PlusAPI(
                api_key=get_auth_token(),
            )
        except AuthError:
            self.plus_api = PlusAPI(api_key="")
        self.ephemeral_trace_url = None

    def initialize_batch(
        self,
        user_context: dict[str, str],
        execution_metadata: dict[str, Any],
        use_ephemeral: bool = False,
    ) -> TraceBatch:
        """Initialize a new trace batch"""
        self.current_batch = TraceBatch(
            user_context=user_context, execution_metadata=execution_metadata
        )
        self.event_buffer.clear()
        self.is_current_batch_ephemeral = use_ephemeral

        self.record_start_time("execution")

        if should_auto_collect_first_time_traces():
            self.trace_batch_id = self.current_batch.batch_id
        else:
            self._initialize_backend_batch(
                user_context, execution_metadata, use_ephemeral
            )
            self.backend_initialized = True

        return self.current_batch

    def _initialize_backend_batch(
        self,
        user_context: dict[str, str],
        execution_metadata: dict[str, Any],
        use_ephemeral: bool = False,
    ):
        """Send batch initialization to backend"""

        if not self.plus_api or not self.current_batch:
            return

        try:
            payload = {
                "trace_id": self.current_batch.batch_id,
                "execution_type": execution_metadata.get("execution_type", "crew"),
                "user_identifier": execution_metadata.get("user_context", None),
                "execution_context": {
                    "crew_fingerprint": execution_metadata.get("crew_fingerprint"),
                    "crew_name": execution_metadata.get("crew_name", None),
                    "flow_name": execution_metadata.get("flow_name", None),
                    "crewai_version": self.current_batch.version,
                    "privacy_level": user_context.get("privacy_level", "standard"),
                },
                "execution_metadata": {
                    "expected_duration_estimate": execution_metadata.get(
                        "expected_duration_estimate", 300
                    ),
                    "agent_count": execution_metadata.get("agent_count", 0),
                    "task_count": execution_metadata.get("task_count", 0),
                    "flow_method_count": execution_metadata.get("flow_method_count", 0),
                    "execution_started_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            if use_ephemeral:
                payload["ephemeral_trace_id"] = self.current_batch.batch_id

            response = (
                self.plus_api.initialize_ephemeral_trace_batch(payload)
                if use_ephemeral
                else self.plus_api.initialize_trace_batch(payload)
            )

            if response is None:
                logger.warning(
                    "Trace batch initialization failed gracefully. Continuing without tracing."
                )
                return

            if response.status_code in [201, 200]:
                response_data = response.json()
                self.trace_batch_id = (
                    response_data["trace_id"]
                    if not use_ephemeral
                    else response_data["ephemeral_trace_id"]
                )
            else:
                logger.warning(
                    f"Trace batch initialization returned status {response.status_code}. Continuing without tracing."
                )

        except Exception as e:
            logger.warning(
                f"Error initializing trace batch: {e}. Continuing without tracing."
            )

    def add_event(self, trace_event: TraceEvent):
        """Add event to buffer"""
        self.event_buffer.append(trace_event)

    def _send_events_to_backend(self) -> int:
        """Send buffered events to backend with graceful failure handling"""
        if not self.plus_api or not self.trace_batch_id or not self.event_buffer:
            return 500
        try:
            payload = {
                "events": [event.to_dict() for event in self.event_buffer],
                "batch_metadata": {
                    "events_count": len(self.event_buffer),
                    "batch_sequence": 1,
                    "is_final_batch": False,
                },
            }

            response = (
                self.plus_api.send_ephemeral_trace_events(self.trace_batch_id, payload)
                if self.is_current_batch_ephemeral
                else self.plus_api.send_trace_events(self.trace_batch_id, payload)
            )

            if response is None:
                logger.warning("Failed to send trace events. Events will be lost.")
                return 500

            if response.status_code in [200, 201]:
                self.event_buffer.clear()
                return 200

            logger.warning(
                f"Failed to send events: {response.status_code}. Events will be lost."
            )
            return 500

        except Exception as e:
            logger.warning(
                f"Error sending events to backend: {e}. Events will be lost."
            )
            return 500

    def finalize_batch(self) -> TraceBatch | None:
        """Finalize batch and return it for sending"""
        if not self.current_batch:
            return None

        self.current_batch.events = self.event_buffer.copy()
        if self.event_buffer:
            events_sent_to_backend_status = self._send_events_to_backend()
            if events_sent_to_backend_status == 500:
                return None
        self._finalize_backend_batch()

        finalized_batch = self.current_batch

        self.batch_owner_type = None
        self.batch_owner_id = None

        self.current_batch = None
        self.event_buffer.clear()
        self.trace_batch_id = None
        self.is_current_batch_ephemeral = False

        self._cleanup_batch_data()

        return finalized_batch

    def _finalize_backend_batch(self):
        """Send batch finalization to backend"""
        if not self.plus_api or not self.trace_batch_id:
            return

        try:
            total_events = len(self.current_batch.events) if self.current_batch else 0

            payload = {
                "status": "completed",
                "duration_ms": self.calculate_duration("execution"),
                "final_event_count": total_events,
            }

            response = (
                self.plus_api.finalize_ephemeral_trace_batch(
                    self.trace_batch_id, payload
                )
                if self.is_current_batch_ephemeral
                else self.plus_api.finalize_trace_batch(self.trace_batch_id, payload)
            )

            if response.status_code == 200:
                access_code = response.json().get("access_code", None)
                console = Console()
                return_link = (
                    f"{CREWAI_BASE_URL}/crewai_plus/trace_batches/{self.trace_batch_id}"
                    if not self.is_current_batch_ephemeral and access_code is None
                    else f"{CREWAI_BASE_URL}/crewai_plus/ephemeral_trace_batches/{self.trace_batch_id}?access_code={access_code}"
                )

                if self.is_current_batch_ephemeral:
                    self.ephemeral_trace_url = return_link

                # Create a properly formatted message with URL on its own line
                message_parts = [
                    f"✅ Trace batch finalized with session ID: {self.trace_batch_id}",
                    "",
                    f"🔗 View here: {return_link}",
                ]

                if access_code:
                    message_parts.append(f"🔑 Access Code: {access_code}")

                panel = Panel(
                    "\n".join(message_parts),
                    title="Trace Batch Finalization",
                    border_style="green",
                )
                if not should_auto_collect_first_time_traces():
                    console.print(panel)

            else:
                logger.error(
                    f"❌ Failed to finalize trace batch: {response.status_code} - {response.text}"
                )

        except Exception as e:
            logger.error(f"❌ Error finalizing trace batch: {e}")
            # TODO: send error to app marking as failed

    def _cleanup_batch_data(self):
        """Clean up batch data after successful finalization to free memory"""
        try:
            if hasattr(self, "event_buffer") and self.event_buffer:
                self.event_buffer.clear()

            if hasattr(self, "current_batch") and self.current_batch:
                if hasattr(self.current_batch, "events") and self.current_batch.events:
                    self.current_batch.events.clear()
                self.current_batch = None

            if hasattr(self, "batch_sequence"):
                self.batch_sequence = 0

        except Exception as e:
            logger.error(f"Warning: Error during cleanup: {e}")

    def has_events(self) -> bool:
        """Check if there are events in the buffer"""
        return len(self.event_buffer) > 0

    def get_event_count(self) -> int:
        """Get number of events in buffer"""
        return len(self.event_buffer)

    def is_batch_initialized(self) -> bool:
        """Check if batch is initialized"""
        return self.current_batch is not None

    def record_start_time(self, key: str):
        """Record start time for duration calculation"""
        self.execution_start_times[key] = datetime.now(timezone.utc)

    def calculate_duration(self, key: str) -> int:
        """Calculate duration in milliseconds from recorded start time"""
        start_time = self.execution_start_times.get(key)
        if start_time:
            duration_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            del self.execution_start_times[key]
            return duration_ms
        return 0

    def get_trace_id(self) -> str | None:
        """Get current trace ID"""
        if self.current_batch:
            return self.current_batch.user_context.get("trace_id")
        return None
