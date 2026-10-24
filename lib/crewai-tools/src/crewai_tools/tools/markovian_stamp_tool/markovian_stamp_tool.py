import asyncio
import hashlib
import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


MARKOVIAN_BASE_URL = "https://api.markovianprotocol.com"


class MarkovianStampToolInput(BaseModel):
    """Input schema for MarkovianStampTool."""

    data: str = Field(
        ...,
        description=(
            "The content to stamp, for example an agent's final answer, a "
            "decision, or any text whose existence you want to prove."
        ),
    )
    label: str | None = Field(
        default=None,
        description="Optional human-readable label to attach to the stamp.",
    )


class MarkovianStampTool(BaseTool):
    """Stamp any data on the Markovian Protocol and return a verifiable receipt.

    The tool hashes the data locally with SHA-256 and sends only that digest
    (plus the optional label) to the Markovian API; the text itself never
    leaves the process. The digest is recorded in the public witnessed
    transparency log and the tool returns a Merkle root, the log index, and a
    public verify URL. Anyone holding the original text can recompute the
    SHA-256 and compare it to the receipt's data_hash. The receipt proves the
    data existed at stamping time, not that the data is correct.

    No account, wallet, or API key is required.
    """

    name: str = "Markovian Stamp"
    description: str = (
        "Create a verifiable provenance receipt for any text on the Markovian "
        "Protocol. Only a SHA-256 hash of the text is sent. Returns a Merkle "
        "root and a public verify URL that anyone can check without an account. "
        "Use it to show an agent output existed at a point in time; it does not "
        "show the output is correct."
    )
    args_schema: type[BaseModel] = MarkovianStampToolInput

    base_url: str = MARKOVIAN_BASE_URL
    timeout: int = 30

    def _stamp(self, data_hash: str, label: str | None) -> dict:
        """POST the digest to the stamp endpoint and return the JSON receipt."""
        resp = requests.post(
            f"{self.base_url}/stamp",
            json={"data_hash": data_hash, "label": label},
            headers={
                "User-Agent": "crewai-tools-markovian",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _run(self, data: str, label: str | None = None, **_: Any) -> str:
        """Hash and stamp the data, returning errors as readable text."""
        data_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
        try:
            receipt = self._stamp(data_hash, label)
        except requests.Timeout:
            return "Markovian stamp timed out. Please try again later."
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            body = exc.response.text[:160] if exc.response is not None else ""
            return f"Markovian stamp failed: HTTP {status} {body}"
        except (requests.RequestException, ValueError) as exc:
            return f"Markovian stamp failed: {exc}"

        if not isinstance(receipt, dict):
            return f"Markovian stamp returned an unexpected response: {receipt}"

        merkle_root = receipt.get("merkle_root")
        if not merkle_root:
            return f"Markovian stamp returned no merkle_root: {json.dumps(receipt)}"

        verify_url = receipt.get("verify_url", f"{self.base_url}/verify/{merkle_root}")
        lines = [
            "Markovian provenance receipt:",
            f"  data_hash: {data_hash} (sha256 of the stamped text)",
            f"  merkle_root: {merkle_root}",
        ]
        if receipt.get("log_index") is not None:
            lines.append(f"  log_index: {receipt['log_index']}")
        lines.append(f"  verify_url: {verify_url}")
        return "\n".join(lines)

    async def _arun(self, data: str, label: str | None = None, **kwargs: Any) -> str:
        """Run the blocking stamp call in a worker thread so the event loop is free."""
        return await asyncio.to_thread(self._run, data, label=label, **kwargs)
