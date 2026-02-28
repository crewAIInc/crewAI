"""Primordia metering for CrewAI agents."""
import hashlib
import json
import time
from typing import Dict, List, Optional


class PrimordiaMeter:
    """Meter that emits MSR receipts for crew task execution.

    Shadow mode by default - no network calls, no blocking.

    Example:
        >>> from crewai.utilities import PrimordiaMeter
        >>> meter = PrimordiaMeter(agent_id="crew-alpha")
        >>> crew = Crew(agents=[...], tasks=[...], meter=meter)
    """

    def __init__(
        self,
        agent_id: str,
        kernel_url: str = "https://clearing.kaledge.app",
    ):
        self.agent_id = agent_id
        self.kernel_url = kernel_url
        self.receipts: List[Dict] = []

    def record_task(
        self,
        task_name: str,
        agent_name: str,
        tokens_used: int,
        model: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> str:
        """Record a task execution as MSR receipt."""
        unit_price = 80  # default
        if "gpt-4" in model.lower():
            unit_price = 300
        elif "claude" in model.lower():
            unit_price = 100

        receipt = {
            "meter_version": "0.1",
            "type": "compute",
            "agent_id": self.agent_id,
            "provider": model,
            "units": tokens_used,
            "unit_price_usd_micros": unit_price,
            "total_usd_micros": tokens_used * unit_price,
            "timestamp_ms": int(time.time() * 1000),
            "metadata": {
                "framework": "crewai",
                "task": task_name,
                "agent": agent_name,
                **(metadata or {})
            }
        }

        receipt_hash = hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()
        ).hexdigest()[:32]

        self.receipts.append({"hash": receipt_hash, "receipt": receipt})
        return receipt_hash

    def get_crew_cost(self) -> float:
        """Get total crew cost in USD."""
        return sum(r["receipt"]["total_usd_micros"] for r in self.receipts) / 1_000_000

    def get_receipts(self) -> List[Dict]:
        """Get all receipts for settlement."""
        return self.receipts
