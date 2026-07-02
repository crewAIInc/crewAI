"""Agent Guild tools — vet another AI agent before delegating work to it.

Agent Guild (https://github.com/AgentTanuki/agent-guild, Apache-2.0) is an
open trust layer for AI agents: an attack-resistant reputation graph
(EigenTrust + collusion detection) over evidence-backed work attestations,
with W3C did:key identity and portable signed reputation credentials.

These tools let a CrewAI agent answer "can I trust this counterparty?" before
handing work — or payment — to an agent it doesn't already know. Reads use the
free/hosted public API; no API key is required for the tools below.
"""

import json
import urllib.parse
import urllib.request
from typing import List, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

AGENT_GUILD_BASE_URL = "https://agent-guild-5d5r.onrender.com"
_UA = "crewai-tools-agentguild/1.0"
_TIMEOUT = 30


def _get(path: str) -> str:
    req = urllib.request.Request(
        AGENT_GUILD_BASE_URL + path, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return r.read().decode("utf-8")


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
    env_vars: List[EnvVar] = []

    def _run(self, capability: str) -> str:
        try:
            return _get("/check?capability=" + urllib.parse.quote(capability))
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"Agent Guild request failed: {e}"})


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
    env_vars: List[EnvVar] = []

    def _run(self, agent_id: str) -> str:
        try:
            return _get(f"/agents/{urllib.parse.quote(agent_id)}/risk-score")
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"Agent Guild request failed: {e}"})


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
    env_vars: List[EnvVar] = []

    def _run(self, credential_json: str) -> str:
        try:
            body = credential_json.encode("utf-8")
            req = urllib.request.Request(
                AGENT_GUILD_BASE_URL + "/credentials/verify", data=body,
                headers={"content-type": "application/json", "User-Agent": _UA},
                method="POST")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"Agent Guild request failed: {e}"})
