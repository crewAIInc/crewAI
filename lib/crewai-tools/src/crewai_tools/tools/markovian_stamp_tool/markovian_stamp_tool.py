import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


MARKOVIAN_BASE_URL = "https://api.quantsynth.net"


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

    Markovian is a content-agnostic provenance primitive. Stamping commits a
    hash of the data to the chain, anchored to Bitcoin, and returns a Merkle
    root plus a public verify URL. It proves the data existed at a point in
    time, not that the data is correct: provenance, not truth.

    No account, wallet, or API key is required. Provide an api_key only for
    attributed or pro usage.
    """

    name: str = "Markovian Stamp"
    description: str = (
        "Create a verifiable provenance receipt for any text on the Markovian "
        "Protocol. Returns a Merkle root and a public verify URL, anchored to "
        "Bitcoin. Use it to prove an agent output existed at a point in time. "
        "No account or API key required."
    )
    args_schema: type[BaseModel] = MarkovianStampToolInput

    package_dependencies: list[str] = Field(default_factory=lambda: ["markovian"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="MARKOVIAN_API_KEY",
                description="Optional API key for attributed or pro usage.",
                required=False,
            ),
        ]
    )

    api_key: str | None = None
    wallet: str | None = None
    base_url: str = MARKOVIAN_BASE_URL
    timeout: int = 30

    def __init__(
        self,
        api_key: str | None = None,
        wallet: str | None = None,
        base_url: str = MARKOVIAN_BASE_URL,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("MARKOVIAN_API_KEY")
        self.wallet = wallet
        self.base_url = base_url
        self.timeout = timeout

    def _stamp(self, data: str, label: str | None) -> dict:
        """Stamp via the markovian SDK when available, else a plain POST."""
        try:
            from markovian import MarkovianClient

            client = MarkovianClient(api_key=self.api_key, timeout=self.timeout)
            return client.stamp(data, wallet=self.wallet, label=label)
        except ImportError:
            payload: dict = {"data": data, "label": label}
            if self.wallet:
                payload["wallet"] = self.wallet
            headers = {
                "User-Agent": "crewai-tools-markovian",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            resp = requests.post(
                f"{self.base_url}/stamp",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

    def _run(self, data: str, label: str | None = None, **_: Any) -> str:
        try:
            receipt = self._stamp(data, label)
        except requests.Timeout:
            return "Markovian stamp timed out. Please try again later."
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            body = exc.response.text[:160] if exc.response is not None else ""
            return f"Markovian stamp failed: HTTP {status} {body}"
        except Exception as exc:
            return f"Markovian stamp failed: {exc}"

        if not isinstance(receipt, dict):
            return f"Markovian stamp returned an unexpected response: {receipt}"

        merkle_root = receipt.get("merkle_root")
        if not merkle_root:
            return f"Markovian stamp returned no merkle_root: {json.dumps(receipt)}"

        verify_url = receipt.get("verify_url", f"{self.base_url}/verify/{merkle_root}")
        block_height = receipt.get("block_height")
        lines = [
            "Markovian provenance receipt (markovian-provenance/v1):",
            f"  merkle_root: {merkle_root}",
            f"  block_height: {block_height}",
            f"  verify_url: {verify_url}",
        ]
        return "\n".join(lines)

    async def _arun(self, data: str, label: str | None = None, **kwargs: Any) -> str:
        return self._run(data, label=label, **kwargs)
