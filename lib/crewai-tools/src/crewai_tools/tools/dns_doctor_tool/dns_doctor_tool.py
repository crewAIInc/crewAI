"""DNS Doctor tools: deterministic DNS and email-authentication checks.

Three tools over the hosted DNS Doctor API (https://dnsdoctor.dev): a full
domain scan, the next safe DMARC record, and a six-location propagation check.
Every verdict is deterministic and every record comes from a validating
engine, never from a language model. Responses are relayed verbatim as JSON
text; nothing is composed here.
"""

from importlib.metadata import PackageNotFoundError, version
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


DEFAULT_API_BASE = "https://dnsdoctor.dev"
_TIMEOUT_SECONDS = 45
_NOT_A_VERDICT = "retry later; this is not a verdict about the domain"

_ENV_VARS: list[EnvVar] = [
    EnvVar(
        name="DNSDOCTOR_API_TOKEN",
        description="Optional DNS Doctor API token. Raises the anonymous rate limit; never required.",
        required=False,
    ),
]


def _user_agent() -> str:
    try:
        tools_version = version("crewai-tools")
    except PackageNotFoundError:
        tools_version = "0"
    return f"dnsdoctor-crewai/{tools_version} (+https://dnsdoctor.dev)"


def _call(path: str, body: dict[str, Any]) -> str:
    """POST one request and return the API body as the API sent it, untouched.

    A transport failure (network, 429, 402, 5xx) returns a message that says it
    is not a verdict about the domain, so an agent never reads it as one.
    """
    base = os.environ.get("DNSDOCTOR_API_BASE", DEFAULT_API_BASE).rstrip("/")
    headers = {"Accept": "application/json", "User-Agent": _user_agent()}
    token = os.environ.get("DNSDOCTOR_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.post(
            f"{base}{path}", json=body, headers=headers, timeout=_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        return f"DNS Doctor could not be reached ({exc}); {_NOT_A_VERDICT}."
    status = response.status_code
    if status == 429:
        return f"DNS Doctor rate-limited this caller; {_NOT_A_VERDICT}."
    if status == 402:
        # The body is the x402 payment offer (the v1 document; the v2 document
        # rides in the PAYMENT-REQUIRED header). Relay both so an x402-capable
        # caller can pay and retry; the text says it is not a verdict.
        offer_header = response.headers.get("PAYMENT-REQUIRED", "")
        return (
            "DNS Doctor: past the free per-caller allowance. The API offered a paid "
            "retry over x402 (USDC on Base); wait and retry, or pay with an x402 client. "
            "This is not a verdict about the domain.\n"
            f"x402 offer (body): {response.text}\n"
            f"x402 offer (PAYMENT-REQUIRED header): {offer_header}"
        )
    if status >= 500:
        return f"DNS Doctor returned HTTP {status} (transient); {_NOT_A_VERDICT}."
    try:
        payload = response.json()
    except ValueError:
        return f"DNS Doctor returned an unreadable response (HTTP {status}); {_NOT_A_VERDICT}."
    if status >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        return f"DNS Doctor refused the request (HTTP {status}): {detail or payload}"
    # Validated above; relayed as the API sent it (no re-serialisation).
    return response.text


class DnsDoctorScanToolSchema(BaseModel):
    """Input for DnsDoctorScanTool."""

    domain: str = Field(
        ..., description="The domain to scan, e.g. example.com (no scheme, no path)."
    )


class DnsDoctorScanTool(BaseTool):
    """Scan a domain's DNS and email-authentication posture."""

    name: str = "DNS Doctor domain scan"
    description: str = (
        "Scan a domain's SPF, DKIM, DMARC, MX, DNS health, blacklist status and domain/TLS "
        "expiry. Returns deterministic per-check verdicts (pass, warn, fail, info, "
        "temperror), plain-English findings and copy-paste fix records from a validating "
        "engine. Present any returned record verbatim. A 'temperror' status is a transient "
        "lookup failure, not a failure of the domain. 'not_registered: true' means the "
        "domain does not resolve and no check ran. SPF is diagnose-only: never propose SPF "
        "edits of your own. A human must approve every DNS change."
    )
    args_schema: type[BaseModel] = DnsDoctorScanToolSchema
    env_vars: list[EnvVar] = _ENV_VARS

    def _run(self, domain: str, **_: Any) -> str:
        return _call("/api/v1/scan", {"domain": domain})


class DnsDoctorDmarcUpgradeToolSchema(BaseModel):
    """Input for DnsDoctorDmarcUpgradeTool."""

    domain: str = Field(..., description="The domain whose DMARC policy to advance.")


class DnsDoctorDmarcUpgradeTool(BaseTool):
    """The next safe DMARC record for a domain."""

    name: str = "DNS Doctor DMARC upgrade"
    description: str = (
        "Build the next safe DMARC record for a domain (none -> quarantine, alignment-gated; "
        "p=reject only with aggregate-report evidence). Returns 'record' (present it verbatim), "
        "'rationale' and the domain's 'current_policy'. 'record' can be null: then the "
        "rationale IS the answer, never compose a record to fill the gap. A human must "
        "approve the DNS change."
    )
    args_schema: type[BaseModel] = DnsDoctorDmarcUpgradeToolSchema
    env_vars: list[EnvVar] = _ENV_VARS

    def _run(self, domain: str, **_: Any) -> str:
        return _call("/api/v1/dmarc-upgrade", {"domain": domain})


class DnsDoctorPropagationToolSchema(BaseModel):
    """Input for DnsDoctorPropagationTool."""

    name: str = Field(..., description="The DNS name to read, e.g. www.example.com.")
    record_type: Literal["A", "AAAA", "CNAME", "MX", "TXT", "NS"] = Field(
        "A", description="Record type to read: A, AAAA, CNAME, MX, TXT or NS."
    )
    expected_value: str | None = Field(
        None,
        description="The value every location should answer with. Omit to check only "
        "that all locations agree.",
    )


class DnsDoctorPropagationTool(BaseTool):
    """Has a DNS change propagated worldwide?"""

    name: str = "DNS Doctor propagation check"
    description: str = (
        "Read one DNS name from six locations on four continents and report whether a "
        "change has propagated. Verdict is 'propagated', 'partial' or 'not_propagated' when "
        "an expected value is given, 'consistent' or 'inconsistent' otherwise, and 'unknown' "
        "when fewer than three locations answered. A location that did not answer is "
        "reported as unreached, never as a negative result. Observation only: no record is "
        "composed."
    )
    args_schema: type[BaseModel] = DnsDoctorPropagationToolSchema
    env_vars: list[EnvVar] = _ENV_VARS

    def _run(
        self,
        name: str,
        record_type: str = "A",
        expected_value: str | None = None,
        **_: Any,
    ) -> str:
        body: dict[str, Any] = {"name": name, "record_type": record_type}
        if expected_value is not None:
            body["expected_value"] = expected_value
        return _call("/api/tools/propagation-check", body)
