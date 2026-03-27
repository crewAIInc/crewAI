"""
SafetyMD Tool for CrewAI
Verify payment addresses before sending funds via the SafetyMD API.
API: https://safetymd.p-u-c.workers.dev/v1/check/{address}?chain={chain}
Free tier: 10 checks/day, no auth required.
"""

from __future__ import annotations

from typing import Any, Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


SAFETYMD_BASE_URL = "https://safetymd.p-u-c.workers.dev/v1/check"
TIMEOUT_SECONDS = 3

RISK_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "critical": "🔴",
}


class SafetyMDInput(BaseModel):
    address: str = Field(
        description="Ethereum-style 0x address to check (e.g. 0xabc...def)"
    )
    chain: str = Field(
        default="base",
        description="Blockchain to check on. Supported: base, ethereum, arbitrum",
    )


class SafetyMDTool(BaseTool):
    name: str = "check_payment_address"
    description: str = (
        "Verify if a payment address is safe before sending funds via MPP or any "
        "blockchain payment. Returns a safety assessment with risk level and reason. "
        "Always call this tool before authorising any fund transfer."
    )
    args_schema: Type[BaseModel] = SafetyMDInput

    def _run(self, address: str, chain: str = "base") -> str:
        """
        Check a payment address against the SafetyMD API.

        Returns a structured human-readable safety assessment string.
        Never raises — all errors are captured and returned as a safe failure message.
        """
        url = f"{SAFETYMD_BASE_URL}/{address}"
        params = {"chain": chain}

        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except requests.exceptions.Timeout:
            return _format_error(address, chain, "Request timed out after 3 seconds. Treat as UNVERIFIED — do not proceed without manual review.")
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            return _format_error(address, chain, f"HTTP {status} from SafetyMD API. Treat as UNVERIFIED.")
        except requests.exceptions.RequestException as exc:
            return _format_error(address, chain, f"Network error: {exc}. Treat as UNVERIFIED.")
        except ValueError:
            return _format_error(address, chain, "Invalid JSON response from SafetyMD API. Treat as UNVERIFIED.")

        return _format_result(address, chain, data)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_result(address: str, chain: str, data: dict[str, Any]) -> str:
    safe: bool = data.get("safe", False)
    risk: str = data.get("risk", "unknown").lower()
    reason: str = data.get("reason", "No reason provided.")
    service: dict[str, Any] = data.get("service", {})
    signals: dict[str, Any] = data.get("signals", {})

    emoji = RISK_EMOJI.get(risk, "⚪")
    verdict = "✅ SAFE TO PROCEED" if safe else "🚫 DO NOT SEND FUNDS"

    lines = [
        f"SafetyMD Check — {address} ({chain})",
        f"Verdict : {verdict}",
        f"Risk    : {emoji} {risk.upper()}",
        f"Reason  : {reason}",
    ]

    if service:
        service_name = service.get("name") or service.get("label") or str(service)
        lines.append(f"Service : {service_name}")

    if signals:
        sig_parts = [f"{k}={v}" for k, v in signals.items()]
        lines.append(f"Signals : {', '.join(sig_parts)}")

    lines.append("")
    if not safe:
        lines.append("⚠️  RECOMMENDATION: Halt payment. Escalate to a human operator for manual review.")
    else:
        lines.append("✔  RECOMMENDATION: Address appears safe. Proceed with normal caution.")

    return "\n".join(lines)


def _format_error(address: str, chain: str, detail: str) -> str:
    return (
        f"SafetyMD Check — {address} ({chain})\n"
        f"Verdict : ⚠️  CHECK FAILED\n"
        f"Detail  : {detail}\n"
        f"\n"
        f"⚠️  RECOMMENDATION: Cannot verify safety. Do not proceed without manual review."
    )
