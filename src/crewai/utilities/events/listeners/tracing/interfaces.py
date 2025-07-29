from abc import ABC, abstractmethod
import json
from typing import Any
from datetime import datetime

from crewai.cli.plus_api import PlusAPI
from crewai.cli.authentication.token import get_auth_token

from .trace_batch_manager import TraceBatch


class ITraceSender(ABC):
    """Interface for sending trace batches (DIP compliance)"""

    @abstractmethod
    def send_batch(self, batch: TraceBatch) -> bool:
        """Send a trace batch to the backend"""
        pass


class ConsoleTraceSender(ITraceSender):
    """Console implementation for testing and development"""

    # sending trace_event_batches to the appropriate session_id
    def send_batch(self, batch: TraceBatch) -> bool:
        """Print trace batch to console"""
        try:
            payload = batch.to_dict()

            def datetime_handler(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat() + "Z" if obj.tzinfo else obj.isoformat()

            # Serialize payload with datetime handler before sending to API
            serialized_payload = json.loads(
                json.dumps(payload, default=datetime_handler)
            )

            PlusAPI(api_key=get_auth_token()).send_trace_batch(
                serialized_payload
            )  # TODO: proper flow
            return True
        except Exception as e:
            print(f"❌ Error sending trace batch: {e}")
            return False


class HttpTraceSender(ITraceSender):
    """HTTP implementation for production use"""

    def __init__(self, endpoint: str, api_key: str, timeout: int = 30):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def send_batch(self, batch: TraceBatch) -> bool:
        """Send trace batch via HTTP"""
        try:
            import requests

            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"CrewAI-Tracer/{batch.version}",
            }

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = batch.to_dict()

            response = requests.post(
                self.endpoint, json=payload, headers=headers, timeout=self.timeout
            )

            response.raise_for_status()
            print(f"✅ Trace batch sent successfully (batch_id: {batch.batch_id})")
            return True

        except ImportError:
            print("❌ requests library not available for HTTP trace sending")
            return False
        except Exception as e:
            print(f"❌ Error sending trace batch via HTTP: {e}")
            return False


class FileTraceSender(ITraceSender):
    """File implementation for local storage"""

    def __init__(self, file_path: str = "traces.jsonl"):
        self.file_path = file_path

    def send_batch(self, batch: TraceBatch) -> bool:
        """Append trace batch to file"""
        try:
            payload = batch.to_dict()

            with open(self.file_path, "a") as f:
                f.write(json.dumps(payload, default=str) + "\n")

            print(
                f"✅ Trace batch saved to {self.file_path} (batch_id: {batch.batch_id})"
            )
            return True

        except Exception as e:
            print(f"❌ Error saving trace batch to file: {e}")
            return False


class IEventCorrelator(ABC):
    """Interface for adding event correlations (future extension)"""

    @abstractmethod
    def add_correlations(self, event: Any, source: Any, event_data: Any) -> Any:
        """Add correlations to an event"""
        pass
