import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from crewai.utilities.constants import CREWAI_BASE_URL
from crewai.cli.authentication.token import AuthError, get_auth_token

from crewai.cli.version import get_crewai_version
from crewai.cli.plus_api import PlusAPI
from rich.console import Console
from rich.panel import Panel

from crewai.utilities.events.listeners.tracing.types import TraceEvent
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class TraceBatch:
    """Batch of events to send to backend"""

    version: str = field(default_factory=get_crewai_version)
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_context: Dict[str, str] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    events: List[TraceEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "batch_id": self.batch_id,
            "user_context": self.user_context,
            "execution_metadata": self.execution_metadata,
            "events": [event.to_dict() for event in self.events],
        }


class TraceBatchManager:
    """Single responsibility: Manage batches and event buffering"""

    is_current_batch_ephemeral: bool = False

    def __init__(self):
        try:
            self.plus_api = PlusAPI(api_key=get_auth_token())
        except AuthError:
            self.plus_api = PlusAPI(api_key="")

        self.trace_batch_id: Optional[str] = None  # Backend ID
        self.current_batch: Optional[TraceBatch] = None
        self.event_buffer: List[TraceEvent] = []
        self.execution_start_times: Dict[str, datetime] = {}

    def initialize_batch(
        self,
        user_context: Dict[str, str],
        execution_metadata: Dict[str, Any],
        use_ephemeral: bool = False,
    ) -> TraceBatch:
        """Initialize a new trace batch"""
        self.current_batch = TraceBatch(
            user_context=user_context, execution_metadata=execution_metadata
        )
        self.event_buffer.clear()
        self.is_current_batch_ephemeral = use_ephemeral

        self.record_start_time("execution")
        self._initialize_backend_batch(user_context, execution_metadata, use_ephemeral)

        return self.current_batch

    def _initialize_backend_batch(
        self,
        user_context: Dict[str, str],
        execution_metadata: Dict[str, Any],
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

            if response.status_code == 201 or response.status_code == 200:
                response_data = response.json()
                self.trace_batch_id = (
                    response_data["trace_id"]
                    if not use_ephemeral
                    else response_data["ephemeral_trace_id"]
                )
                console = Console()
                panel = Panel(
                    f"✅ Trace batch initialized with session ID: {self.trace_batch_id}",
                    title="Trace Batch Initialization",
                    border_style="green",
                )
                console.print(panel)
            else:
                logger.error(
                    f"❌ Failed to initialize trace batch: {response.status_code} - {response.text}"
                )

        except Exception as e:
            logger.error(f"❌ Error initializing trace batch: {str(e)}")

    def add_event(self, trace_event: TraceEvent):
        """Add event to buffer"""
        self.event_buffer.append(trace_event)

    def _send_events_to_backend(self):
        """Send buffered events to backend"""
        if not self.plus_api or not self.trace_batch_id or not self.event_buffer:
            return

        try:
            payload = {
                "events": [event.to_dict() for event in self.event_buffer],
                "batch_metadata": {
                    "events_count": len(self.event_buffer),
                    "batch_sequence": 1,
                    "is_final_batch": False,
                },
            }

            if not self.trace_batch_id:
                raise Exception("❌ Trace batch ID not found")

            response = (
                self.plus_api.send_ephemeral_trace_events(self.trace_batch_id, payload)
                if self.is_current_batch_ephemeral
                else self.plus_api.send_trace_events(self.trace_batch_id, payload)
            )

            if response.status_code == 200 or response.status_code == 201:
                self.event_buffer.clear()
            else:
                logger.error(
                    f"❌ Failed to send events: {response.status_code} - {response.text}"
                )

        except Exception as e:
            logger.error(f"❌ Error sending events to backend: {str(e)}")

    def finalize_batch(self) -> Optional[TraceBatch]:
        """Finalize batch and return it for sending"""
        if not self.current_batch:
            return None

        if self.event_buffer:
            self._send_events_to_backend()
        self._finalize_backend_batch()

        self.current_batch.events = self.event_buffer.copy()

        finalized_batch = self.current_batch

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
                panel = Panel(
                    f"✅ Trace batch finalized with session ID: {self.trace_batch_id}. View here: {return_link} {f', Access Code: {access_code}' if access_code else ''}",
                    title="Trace Batch Finalization",
                    border_style="green",
                )
                console.print(panel)

            else:
                logger.error(
                    f"❌ Failed to finalize trace batch: {response.status_code} - {response.text}"
                )

        except Exception as e:
            logger.error(f"❌ Error finalizing trace batch: {str(e)}")
            # TODO: send error to app

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
            logger.error(f"Warning: Error during cleanup: {str(e)}")

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

    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        if self.current_batch:
            return self.current_batch.user_context.get("trace_id")
        return None
