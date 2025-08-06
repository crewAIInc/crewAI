import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from crewai.cli.authentication.constants import CREWAI_BASE_URL
from crewai.cli.authentication.token import get_auth_token

from crewai.cli.version import get_crewai_version
from crewai.cli.plus_api import PlusAPI
from rich.console import Console
from rich.panel import Panel
from pprint import pprint

from crewai.utilities.events.listeners.tracing.types import TraceEvent


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

    def __init__(self):
        self.plus_api = PlusAPI(api_key=get_auth_token())
        self.trace_batch_id: Optional[str] = None  # Backend ID
        self.current_batch: Optional[TraceBatch] = None
        self.event_buffer: List[TraceEvent] = []
        self.execution_start_times: Dict[str, datetime] = {}

    def initialize_batch(
        self, user_context: Dict[str, str], execution_metadata: Dict[str, Any]
    ) -> TraceBatch:
        """Initialize a new trace batch"""
        # 1. Create local batch
        self.current_batch = TraceBatch(
            user_context=user_context, execution_metadata=execution_metadata
        )
        self.event_buffer.clear()

        self.record_start_time("execution")

        self._initialize_backend_batch(user_context, execution_metadata)

        return self.current_batch

    def _initialize_backend_batch(
        self, user_context: Dict[str, str], execution_metadata: Dict[str, Any]
    ):
        """Send batch initialization to backend"""
        if not self.plus_api or not self.current_batch:
            return

        try:
            print("lorenze made it here 1 before sending payload")
            # Build payload according to plan_api.md structure
            payload = {
                "trace_id": self.current_batch.batch_id,
                "execution_type": execution_metadata.get("execution_type", "crew"),
                "execution_context": {
                    "crew_fingerprint": execution_metadata.get("crew_fingerprint"),
                    "crew_name": execution_metadata.get("crew_name", "Unknown Crew"),
                    "flow_name": execution_metadata.get("flow_name"),  # nullable
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

            response = self.plus_api.initialize_trace_batch(payload)

            if response.status_code == 201 or response.status_code == 200:
                response_data = response.json()
                self.trace_batch_id = response_data[
                    "trace_id"
                ]  # Backend-generated session ID
                print(
                    f"✅ Trace batch initialized with session ID: {self.trace_batch_id}"
                )
                console = Console()
                panel = Panel(
                    f"✅ Trace batch initialized with session ID: {self.trace_batch_id}",
                    title="Trace Batch Initialization",
                    border_style="green",
                )
                console.print(panel)
            else:
                print(
                    f"❌ Failed to initialize trace batch: {response.status_code} - {response.text}"
                )

        except Exception as e:
            print(f"❌ Error initializing trace batch: {str(e)}")
            # Continue without backend tracing if initialization fails

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
            print("payload sending over")
            pprint(payload)
            if not self.trace_batch_id:
                raise Exception("❌ Trace batch ID not found")

            response = self.plus_api.send_trace_events(self.trace_batch_id, payload)

            if response.status_code == 200 or response.status_code == 201:
                print(f"✅ Sent {len(self.event_buffer)} events to backend")
                self.event_buffer.clear()
            else:
                print(
                    f"❌ Failed to send events: {response.status_code} - {response.text}"
                )

        except Exception as e:
            print(f"❌ Error sending events to backend: {str(e)}")

    def finalize_batch(self) -> Optional[TraceBatch]:
        """Finalize batch and return it for sending"""
        if not self.current_batch:
            return None

        # Send any remaining events to backend first
        if self.event_buffer:
            print(f"sending events to backend: {self.event_buffer}")
            self._send_events_to_backend()

        # Send finalization to backend (but don't cleanup yet)
        self._finalize_backend_batch()

        # Copy events to batch for local return (current_batch still exists)
        self.current_batch.events = self.event_buffer.copy()

        # Prepare batch for return
        finalized_batch = self.current_batch

        # Clear state for next batch
        self.current_batch = None
        self.event_buffer.clear()
        self.trace_batch_id = None

        # Now do the cleanup after we're done using current_batch
        self._cleanup_batch_data()

        return finalized_batch

    def _finalize_backend_batch(self):
        """Send batch finalization to backend"""
        if not self.plus_api or not self.trace_batch_id:
            return

        try:
            # Calculate metrics from current batch
            total_events = len(self.current_batch.events) if self.current_batch else 0

            payload = {
                "status": "completed",
                "duration_ms": self.calculate_duration("execution"),
                "final_event_count": total_events,
            }

            response = self.plus_api.finalize_trace_batch(self.trace_batch_id, payload)

            if response.status_code == 200:
                print(f"✅ Trace batch finalized successfully")
                console = Console()
                panel = Panel(
                    f"✅ Trace batch finalized with session ID: {self.trace_batch_id}. View here: {CREWAI_BASE_URL}/crewai_plus/trace_batches/{self.trace_batch_id}",
                    title="Trace Batch Finalization",
                    border_style="green",
                )
                console.print(panel)

            else:
                print(
                    f"❌ Failed to finalize trace batch: {response.status_code} - {response.text}"
                )

        except Exception as e:
            print(f"❌ Error finalizing trace batch: {str(e)}")
            # send error to backend
            # self.plus_api.send_error_to_backend(self.trace_batch_id, payload)

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

            print(f"🧹 Cleaned up trace batch data")

        except Exception as e:
            print(f"⚠️ Warning: Error during cleanup: {str(e)}")

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
        print(f"⚠️ Warning: No start time recorded for key: {key}")
        return 0

    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        if self.current_batch:
            return self.current_batch.user_context.get("trace_id")
        return None
