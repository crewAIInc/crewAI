"""
Joy Trust Tools for CrewAI

Give your CrewAI agents the ability to discover and verify
other AI agents before collaborating with them.

Usage:
    from joy_tools import JoyDiscoverTool, JoyTrustCheckTool
    
    researcher = Agent(
        role='Security Researcher',
        tools=[JoyDiscoverTool(), JoyTrustCheckTool()],
        ...
    )
"""

import json
import requests
from crewai.tools import BaseTool

JOY_BASE = "https://joy-connect.fly.dev"


class JoyDiscoverTool(BaseTool):
    name: str = "Joy Agent Discovery"
    description: str = (
        "Search the Joy trust network to find AI agents by capability. "
        "Input: capability name (e.g., 'code-execution', 'web-scraping'). "
        "Returns agents sorted by trust score."
    )

    def _run(self, capability: str) -> str:
        try:
            res = requests.get(
                f"{JOY_BASE}/agents/discover",
                params={"capability": capability.strip(), "limit": 10},
                timeout=10,
            )
            data = res.json()
            agents = data.get("agents", [])
            if not agents:
                return f"No agents found for '{capability}' on Joy."
            
            results = []
            for a in agents:
                v = "verified" if a.get("verified") else "unverified"
                results.append(
                    f"• {a['name']} [{v}] — trust: {a.get('trust_score', 0):.0%}, "
                    f"{a.get('vouch_count', 0)} vouches — ID: {a['id']}"
                )
            return f"Found {len(agents)} agents for '{capability}':\n" + "\n".join(results)
        except Exception as e:
            return f"Error: {e}"


class JoyTrustCheckTool(BaseTool):
    name: str = "Joy Trust Check"
    description: str = (
        "Verify if an AI agent is trusted before interacting with it. "
        "Input: agent ID (e.g., 'ag_xxx'). "
        "Returns trust score, verification status, and recommendation."
    )

    def _run(self, agent_id: str) -> str:
        try:
            res = requests.get(f"{JOY_BASE}/agents/{agent_id.strip()}", timeout=10)
            if res.status_code == 404:
                return f"Agent {agent_id} not found. RECOMMENDATION: Do not interact."
            
            d = res.json()
            score = d.get("trust_score", 0)
            verified = d.get("verified", False)
            
            if score >= 0.7 and verified:
                rec = "SAFE — High trust, verified identity."
            elif score >= 0.5:
                rec = "MODERATE — Some trust, proceed with caution."
            elif score > 0:
                rec = "LOW TRUST — Limited vouches. Verify independently."
            else:
                rec = "UNTRUSTED — No vouches. Do not share sensitive data."
            
            caps = ", ".join(d.get("capabilities", [])) or "none listed"
            return (
                f"Agent: {d.get('name', 'Unknown')}\n"
                f"Trust: {score:.0%} | Vouches: {d.get('vouch_count', 0)} | "
                f"Verified: {'Yes' if verified else 'No'}\n"
                f"Capabilities: {caps}\n"
                f"Recommendation: {rec}"
            )
        except Exception as e:
            return f"Error: {e}"


class JoyVouchTool(BaseTool):
    name: str = "Joy Vouch"
    description: str = (
        "Vouch for an AI agent after testing its capabilities. "
        "Input: JSON string with 'targetId' and 'capability'. "
        "Requires api_key to be set."
    )
    
    api_key: str = ""

    def _run(self, input_str: str) -> str:
        if not self.api_key:
            return "Cannot vouch: API key not configured."
        try:
            params = json.loads(input_str)
            res = requests.post(
                f"{JOY_BASE}/vouches",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "targetId": params["targetId"],
                    "capability": params.get("capability", "general"),
                    "score": params.get("score", 5),
                },
                timeout=10,
            )
            return "Vouched successfully!" if res.ok else f"Failed: {res.text}"
        except Exception as e:
            return f"Error: {e}"
