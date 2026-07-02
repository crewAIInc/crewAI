"""Agent Guild tools — vet another AI agent before delegating work to it.

Agent Guild (https://github.com/AgentTanuki/agent-guild, Apache-2.0) is an
open trust layer for AI agents: an attack-resistant reputation graph
(EigenTrust + collusion detection) over evidence-backed work attestations,
with W3C did:key identity and portable signed reputation credentials.

These tools let a CrewAI agent answer "can I trust this counterparty?" before
handing work — or payment — to an agent it doesn't already know. Reads use the
free/hosted public API; no API key is required for the tools below. Set the
``AGENT_GUILD_BASE_URL`` environment variable to point at a self-hosted or
staging instance instead of the hosted default.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

DEFAULT_AGENT_GUILD_BASE_URL = "https://agent-guild-5d5r.onrender.com"
_UA = "crewai-tools-agentguild/1.0"
_TIMEOUT = 30

_ENV_VARS: List[EnvVar] = [
    EnvVar(
        name="AGENT_GUILD_BASE_URL",
        description="Optional override for the Agent Guild API base URL "
                    "(defaults to the hosted instance)",
        required=False,
    ),
]


def _base_url() -> str:
    return os.environ.get(
        "AGENT_GUILD_BASE_URL", DEFAULT_AGENT_GUILD_BASE_URL).rstrip("/")


def _request(path: str, data: Optional[bytes] = None) -> str:
    """Call the Agent Guild API and return the response body as a string.

    Never raises, so a failed lookup can't crash a crew:
    - HTTP error responses return the API's own (JSON) error body, which
      carries a more actionable message than the bare status line;
    - transport failures (DNS, timeout, cold start of the hosted instance)
      return a structured JSON error string that names the endpoint and
      distinguishes "service unreachable" from an in-band API error.
    """
    base_url = _base_url()
    headers = {"User-Agent": _UA}
    if data is not None:
        headers["content-type"] = "application/json"
    req = urllib.request.Request(
        base_url + path, data=data, headers=headers,
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if body:
            return body
        return json.dumps({
            "error": "agent_guild_http_error",
            "status": e.code,
            "detail": str(e),
        })
    except Exception as e:  # noqa: BLE001
        return json.dumps({
            "error": "agent_guild_unreachable",
            "detail": str(e),
            "endpoint": base_url,
            "hint": "The hosted instance may be cold-starting; retry once, "
                    "or set AGENT_GUILD_BASE_URL to a self-hosted instance.",
        })


class AgentGuildCheckInput(BaseModel):
    capability: str = Field(
        ...,
        description="The capability to vet before delegating, e.g. "
                    "'fact-check', 'code-review', 'summarization'")


class AgentGuildCheckTool(BaseTool):
    name: str = "Agent Guild capability check"
    description: str = (
        "Vet a capability before delegating work to another AI agent. Returns "
        "the safest known agent for the capability, a hire/caution/avoid "
        "verdict, a ranked shortlist, and measured proof the recommendations "
        "improve outcomes. Use BEFORE trusting an agent you don't know. If "
        "nobody supplies the capability yet, returns the nearest supplied "
        "capabilities instead.")
    args_schema: Type[BaseModel] = AgentGuildCheckInput
    package_dependencies: List[str] = []
    env_vars: List[EnvVar] = Field(default_factory=lambda: list(_ENV_VARS))

    def _run(self, capability: str) -> str:
        return _request("/check?capability=" + urllib.parse.quote(capability))


class AgentGuildRiskScoreInput(BaseModel):
    agent_id: str = Field(
        ..., description="The Agent Guild agent id to assess, e.g. 'agent_1a2b3c'")


class AgentGuildRiskScoreTool(BaseTool):
    name: str = "Agent Guild risk score"
    description: str = (
        "Get a hire/caution/avoid risk verdict for one specific Agent Guild "
        "agent id, including its trust score and collusion suspicion.")
    args_schema: Type[BaseModel] = AgentGuildRiskScoreInput
    package_dependencies: List[str] = []
    env_vars: List[EnvVar] = Field(default_factory=lambda: list(_ENV_VARS))

    def _run(self, agent_id: str) -> str:
        return _request(f"/agents/{urllib.parse.quote(agent_id)}/risk-score")


class AgentGuildVerifyPassportInput(BaseModel):
    credential_json: str = Field(
        ...,
        description="The Agent Passport (a Guild-signed W3C Verifiable "
                    "Credential) presented by another agent, as a JSON string")


class AgentGuildVerifyPassportTool(BaseTool):
    name: str = "Agent Guild passport verification"
    description: str = (
        "Verify an Agent Passport (Guild-signed W3C Verifiable Credential) "
        "that another agent presented to prove its reputation. Returns "
        "validity plus the subject's CURRENT trust score, so a stale or "
        "forged credential can't mislead.")
    args_schema: Type[BaseModel] = AgentGuildVerifyPassportInput
    package_dependencies: List[str] = []
    env_vars: List[EnvVar] = Field(default_factory=lambda: list(_ENV_VARS))

    def _run(self, credential_json: str) -> str:
        return _request(
            "/credentials/verify", data=credential_json.encode("utf-8"))
