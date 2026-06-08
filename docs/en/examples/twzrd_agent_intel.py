"""
TWZRD Agent Intel tool for CrewAI agents.

Enables CrewAI agents to verify trust scores before authorizing x402
micropayments or interacting with untrusted external agents.

The TWZRD Agent Intel MCP server provides:
  - score_agent(wallet)       — 0-100 trust score + risk flags (free)
  - preflight_check(wallet)   — PASS/FAIL gate for x402 payments (free)
  - get_trust_receipt(wallet) — signed on-chain trust receipt (HTTP 402 paid)

MCP endpoint: https://intel.twzrd.xyz/mcp  (streamable-http, no auth required)
Website: https://intel.twzrd.xyz

Install:
    pip install crewai crewai-tools mcp

Usage:
    python docs/en/examples/twzrd_agent_intel.py
"""
import asyncio
import json

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

TWZRD_MCP_URL = "https://intel.twzrd.xyz/mcp"


# --- Pydantic schemas for tool inputs ---

class ScoreAgentInput(BaseModel):
    wallet: str = Field(..., description="Solana wallet address (base58) of the agent to check")


class PreflightInput(BaseModel):
    wallet: str = Field(..., description="Solana wallet address (base58) to run preflight on")


# --- CrewAI tools wrapping TWZRD MCP ---

class ScoreAgentTool(BaseTool):
    name: str = "score_agent"
    description: str = (
        "Check the TWZRD trust score (0-100) for a Solana agent wallet. "
        "Returns trust score and behavioral risk flags. "
        "Use before authorizing x402 micropayments or delegating to external agents. "
        "Score >= 60 is required for payment authorization."
    )
    args_schema: type[BaseModel] = ScoreAgentInput

    def _run(self, wallet: str) -> str:
        return asyncio.run(self._async_run(wallet))

    async def _async_run(self, wallet: str) -> str:
        async with streamablehttp_client(TWZRD_MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("score_agent", {"wallet": wallet})

        text = result.content[0].text if result.content else "{}"
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            data = {"score": 0, "risk_flags": ["parse_error"]}

        score = data.get("score", 0)
        flags = data.get("risk_flags", [])
        trust_level = "HIGH" if score >= 80 else "MEDIUM" if score >= 60 else "LOW"

        return json.dumps({
            "wallet": wallet,
            "score": score,
            "trust_level": trust_level,
            "risk_flags": flags,
            "authorized_for_payments": score >= 60,
        })


class PreflightCheckTool(BaseTool):
    name: str = "preflight_check"
    description: str = (
        "Run a PASS/FAIL preflight check on a Solana agent wallet. "
        "Returns PASS if the agent is cleared for x402 micropayments, FAIL otherwise. "
        "Always run this before authorizing any payment-sensitive actions."
    )
    args_schema: type[BaseModel] = PreflightInput

    def _run(self, wallet: str) -> str:
        return asyncio.run(self._async_run(wallet))

    async def _async_run(self, wallet: str) -> str:
        async with streamablehttp_client(TWZRD_MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("preflight_check", {"wallet": wallet})

        text = result.content[0].text if result.content else ""
        passed = "PASS" in text.upper()
        return "PASS" if passed else "FAIL"


# --- CrewAI setup ---

def create_trust_verification_crew(agent_wallet: str):
    """Create a CrewAI crew that verifies agent trust before proceeding."""

    trust_tools = [ScoreAgentTool(), PreflightCheckTool()]

    trust_verifier = Agent(
        role="Trust Verification Specialist",
        goal=(
            "Verify the trust score of agent wallets before authorizing x402 "
            "micropayments. Reject any agent with trust score < 60 or preflight FAIL."
        ),
        backstory=(
            "You are a security specialist in the x402 agentic payment ecosystem. "
            "You use on-chain behavioral analysis to protect systems from untrusted agents."
        ),
        tools=trust_tools,
        verbose=True,
    )

    verify_task = Task(
        description=(
            f"Verify the trust score for agent wallet: {agent_wallet}\n"
            "1. Use score_agent to get the trust score and risk flags\n"
            "2. Use preflight_check to get the binary PASS/FAIL result\n"
            "3. Summarize the findings and provide a final AUTHORIZE or REJECT decision\n"
            "   - AUTHORIZE if score >= 60 AND preflight = PASS\n"
            "   - REJECT otherwise, with specific reason"
        ),
        expected_output=(
            "A trust verification report with: trust score, trust level, "
            "risk flags, preflight result, and final AUTHORIZE or REJECT decision."
        ),
        agent=trust_verifier,
    )

    crew = Crew(
        agents=[trust_verifier],
        tasks=[verify_task],
        verbose=True,
    )
    return crew


if __name__ == "__main__":
    # Example: verify an agent wallet before authorizing x402 payment access
    wallet = "4LkEFjHsF2ubC8K4oF2r3rCFqPZQVGBjL9mV6xkNPZdf"

    crew = create_trust_verification_crew(wallet)
    result = crew.kickoff()
    print(f"\nFinal decision: {result}")
