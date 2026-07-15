import os
from typing import Any

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel

from crewai_tools.tools.relayshield_tool.schemas import (
    MCPRegistryRiskParams,
    PromptInjectionBreachParams,
)

API_BASE_URL: str = "https://api.relayshield.net"


def _relayshield_headers(api_key: str) -> dict[str, str]:
    # Note: relayshield_agentic_api.py (which hosts these two endpoints)
    # expects X-RS-API-KEY specifically, not the X-API-Key convention used
    # by RelayShield's other metered endpoints.
    return {"Content-Type": "application/json", "X-RS-API-KEY": api_key}


class RelayShieldMCPRiskTool(BaseTool):
    """Typosquat / reputation / registration-age risk check for MCP servers
    and agent tool registries, backed by RelayShield's live threat-intel API."""

    name: str = "RelayShield MCP Registry Risk"
    description: str = (
        "Checks an MCP server URL or package name for typosquat risk against known "
        "MCP ecosystem domains, presence in RelayShield's criminal IOC corpus, and "
        "domain-registration age. Use this before connecting an agent to an unfamiliar "
        "MCP server or tool registry. Returns a verdict (CRITICAL/HIGH/MEDIUM/LOW) with findings."
    )
    args_schema: type[BaseModel] = MCPRegistryRiskParams
    env_vars: list[EnvVar] = [
        EnvVar(
            name="RELAYSHIELD_API_KEY",
            description="API key for RelayShield (get one at api.relayshield.net/developers)",
            required=True,
        ),
    ]

    def _run(self, server_url: str | None = None, package_name: str | None = None, **kwargs: Any) -> str:
        if not server_url and not package_name:
            return "Error: provide either server_url or package_name."

        api_key = os.environ.get("RELAYSHIELD_API_KEY", "")
        if not api_key:
            return (
                "Error: RELAYSHIELD_API_KEY environment variable is required. "
                "Get a key at https://api.relayshield.net/developers"
            )

        payload: dict[str, str] = {}
        if server_url:
            payload["server_url"] = server_url
        if package_name:
            payload["package_name"] = package_name

        try:
            resp = requests.post(
                f"{API_BASE_URL}/v1/metered/mcp-registry-risk",
                json=payload,
                headers=_relayshield_headers(api_key),
                timeout=15,
            )
            resp.raise_for_status()
            # Every RelayShield response is wrapped as {"ok": bool, "data": {...}}.
            data = resp.json().get("data", {})
        except requests.RequestException as exc:
            return f"RelayShield MCP registry risk check failed: {exc}"

        verdict = data.get("verdict", "UNKNOWN")
        findings = data.get("findings", [])
        if not findings:
            return f"Verdict: {verdict}. No red flags found. {data.get('note', '')}"

        lines = [f"Verdict: {verdict}", "Findings:"]
        for f in findings:
            lines.append(f"- [{f.get('severity')}] {f.get('type')}: {f.get('detail')}")
        return "\n".join(lines)


class RelayShieldPromptInjectionBreachTool(BaseTool):
    """Checks whether an email's credentials were exposed via a breach specifically
    sourced from a prompt-injection attack against an AI agent, as opposed to
    traditional phishing/malware-sourced breaches."""

    name: str = "RelayShield Prompt-Injection Breach Check"
    description: str = (
        "Checks an email address for credential exposure sourced specifically from "
        "prompt-injection attacks against AI agents (distinct from ordinary breach/phishing "
        "sources). Use this to vet an agent identity or user account before granting it "
        "elevated trust or access."
    )
    args_schema: type[BaseModel] = PromptInjectionBreachParams
    env_vars: list[EnvVar] = [
        EnvVar(
            name="RELAYSHIELD_API_KEY",
            description="API key for RelayShield (get one at api.relayshield.net/developers)",
            required=True,
        ),
    ]

    def _run(self, email: str, **kwargs: Any) -> str:
        api_key = os.environ.get("RELAYSHIELD_API_KEY", "")
        if not api_key:
            return (
                "Error: RELAYSHIELD_API_KEY environment variable is required. "
                "Get a key at https://api.relayshield.net/developers"
            )

        try:
            resp = requests.post(
                f"{API_BASE_URL}/v1/metered/prompt-injection-breach",
                json={"email": email},
                headers=_relayshield_headers(api_key),
                timeout=15,
            )
            resp.raise_for_status()
            # Every RelayShield response is wrapped as {"ok": bool, "data": {...}}.
            data = resp.json().get("data", {})
        except requests.RequestException as exc:
            return f"RelayShield prompt-injection breach check failed: {exc}"

        if not data.get("found"):
            return f"No prompt-injection-sourced exposure found for {email}. {data.get('note', '')}"

        count = data.get("session_count", 0)
        return (
            f"Found {count} session(s) exposed via prompt-injection-sourced breach for {email}. "
            "Treat this identity as compromised until credentials are rotated."
        )
