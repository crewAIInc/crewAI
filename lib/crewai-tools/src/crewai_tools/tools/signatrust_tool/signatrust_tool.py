"""
Signatrust tool for CrewAI.

Wraps the Signatrust REST API so agents can generate, verify, and retrieve
cryptographically signed **AI Decision Receipts** — tamper-evident records of
decisions made by AI agents.

Learn more: https://signatrust.net
Source:     https://github.com/abokenan444/Signatrust
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://signatrust.net/api/v1"
DEFAULT_TIMEOUT = 30


class SignatrustToolInput(BaseModel):
    """Input schema for :class:`SignatrustTool`."""

    operation: str = Field(
        "generate",
        description=(
            "Operation to perform. One of: 'generate' (create a new Decision "
            "Receipt), 'verify' (verify an existing receipt by id), or 'get' "
            "(retrieve a receipt by id)."
        ),
    )
    agent_name: Optional[str] = Field(
        None,
        description="Name/identifier of the AI agent making the decision (generate).",
    )
    action: Optional[str] = Field(
        None,
        description="The action being recorded, e.g. 'approve_transaction' (generate).",
    )
    decision: Optional[str] = Field(
        None,
        description="The decision/outcome to record, e.g. 'approved' (generate).",
    )
    model: Optional[str] = Field(
        None,
        description="Model that produced the decision, e.g. 'gpt-4o' (generate).",
    )
    policies: Optional[List[str]] = Field(
        None,
        description="Policies/rules applied while making the decision (generate).",
    )
    human_review: Optional[bool] = Field(
        None,
        description="Whether a human reviewed/approved the decision (generate).",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional structured context (generate).",
    )
    receipt_id: Optional[str] = Field(
        None,
        description="Receipt id, required for the 'verify' and 'get' operations.",
    )


class SignatrustTool(BaseTool):
    """Generate, verify, and retrieve Signatrust AI Decision Receipts.

    Signatrust produces tamper-evident, cryptographically signed (Ed25519)
    receipts for decisions made by AI agents, enabling verifiable accountability
    and auditability. By default only a SHA-256 hash of the decision payload is
    stored server-side (privacy-first).

    Requires the ``SIGNATRUST_API_KEY`` environment variable (or pass
    ``api_key=`` directly). Self-hosted deployments can override ``base_url``.
    """

    name: str = "Signatrust Decision Receipt"
    description: str = (
        "Create, verify, or retrieve cryptographically signed AI Decision "
        "Receipts via Signatrust. Use 'generate' to record an agent decision, "
        "'verify' to check a receipt's integrity, and 'get' to fetch a receipt "
        "by id. Returns a JSON string."
    )
    args_schema: Type[BaseModel] = SignatrustToolInput

    env_vars: List[EnvVar] = [
        EnvVar(
            name="SIGNATRUST_API_KEY",
            description="API key for the Signatrust service (sk_live_...).",
            required=True,
        ),
    ]
    package_dependencies: List[str] = ["requests"]

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    timeout: int = DEFAULT_TIMEOUT

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Lazy dependency check to keep the base install light.
        try:
            import requests  # noqa: F401
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Missing optional dependency 'requests'. Install with:\n"
                "  pip install requests\n"
            ) from exc

        if not self.api_key:
            self.api_key = os.environ.get("SIGNATRUST_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A Signatrust API key is required. Set the SIGNATRUST_API_KEY "
                "environment variable or pass api_key= to SignatrustTool."
            )

    # -- internal helpers -------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.api_key or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        body = {k: v for k, v in payload.items() if v is not None}
        resp = requests.post(
            f"{self.base_url}/receipts",
            headers=self._headers(),
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _verify(self, receipt_id: str) -> Dict[str, Any]:
        import requests

        resp = requests.get(
            f"{self.base_url}/receipts/{receipt_id}/verify",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, receipt_id: str) -> Dict[str, Any]:
        import requests

        resp = requests.get(
            f"{self.base_url}/receipts/{receipt_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # -- execution --------------------------------------------------------

    def _run(self, **kwargs: Any) -> str:
        operation = (kwargs.get("operation") or "generate").strip().lower()

        try:
            if operation == "generate":
                result = self._generate(
                    {
                        "agent_name": kwargs.get("agent_name"),
                        "action": kwargs.get("action"),
                        "decision": kwargs.get("decision"),
                        "model": kwargs.get("model"),
                        "policies": kwargs.get("policies"),
                        "human_review": kwargs.get("human_review"),
                        "metadata": kwargs.get("metadata"),
                    }
                )
            elif operation in {"verify", "get"}:
                receipt_id = kwargs.get("receipt_id")
                if not receipt_id:
                    return (
                        f"Error: 'receipt_id' is required for the '{operation}' "
                        "operation."
                    )
                result = (
                    self._verify(receipt_id)
                    if operation == "verify"
                    else self._get(receipt_id)
                )
            else:
                return (
                    f"Error: unknown operation '{operation}'. Use 'generate', "
                    "'verify', or 'get'."
                )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - return friendly message
            return f"Signatrust request failed: {exc}"

    async def _arun(self, **kwargs: Any) -> str:
        # The underlying client is synchronous and thread-safe; delegate.
        return self._run(**kwargs)
