import json
from datetime import datetime

from crewai.cli.plus_api import PlusAPI
from crewai.cli.authentication.token import get_auth_token
from pydantic import BaseModel
from .trace_batch_manager import TraceBatch
from logging import getLogger

logger = getLogger(__name__)


class TraceSender(BaseModel):
    """Trace sender for sending trace batches to the backend"""

    def send_batch(self, batch: TraceBatch) -> bool:
        """Print trace batch to console"""
        try:
            payload = batch.to_dict()

            def datetime_handler(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat() + "Z" if obj.tzinfo else obj.isoformat()

            serialized_payload = json.loads(
                json.dumps(payload, default=datetime_handler)
            )

            PlusAPI(api_key=get_auth_token()).send_trace_batch(serialized_payload)
            return True
        except Exception as e:
            logger.error(f"Error sending trace batch: {e}")
            return False
