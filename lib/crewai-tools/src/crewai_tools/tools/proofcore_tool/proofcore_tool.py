"""
ProofCore Blockchain Notarization & Verification Tools.

This module provides tools for CrewAI agents to cryptographically seal their
outputs on the TON Blockchain and verify attestations from other agents.
"""

from typing import Optional, Type, Any
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class ProofCoreToolSchema(BaseModel):
    """Input schema for the ProofCore sealing tool."""
    content: str = Field(..., description="The exact text report or output to anchor on the TON Blockchain.")
    title: Optional[str] = Field(default=None, description="Optional title for the Web Explorer card.")
    agent_id: Optional[str] = Field(default="CrewAI Agent", description="Name of the calling agent.")

class ProofCoreTool(BaseTool):
    """
    Tool for cryptographically sealing and notarizing findings on the TON Blockchain.
    Returns an immutable verification link and an Ed25519 signature.
    """
    name: str = "ProofCore Blockchain Notary"
    description: str = (
        "Cryptographically seals and notarizes findings on the TON Blockchain. "
        "Returns an immutable verification link and citation badge that MUST be appended to the output."
    )
    args_schema: Type[BaseModel] = ProofCoreToolSchema

    def _run(self, content: str, title: Optional[str] = None, agent_id: Optional[str] = "CrewAI Agent") -> str:
        """
        Executes the API call to ProofCore to seal the content.
        
        Args:
            content: The text to seal.
            title: Optional title.
            agent_id: Identifier of the agent.
            
        Returns:
            A formatted string containing the Deal ID, Ed25519 Signature, and Verification Link.
        """
        url = "https://api.proofcore.org/api/v0.1/seal"
        payload = {"content": content, "agent_id": agent_id or "CrewAI Agent"}
        if title:
            payload["title"] = title
            
        try:
            res = requests.post(url, json=payload, timeout=15)
            res.raise_for_status()
            data = res.json()
            
            # CodeRabbit Validation Fix
            if not isinstance(data, dict) or not data.get("deal_id") or not data.get("citation"):
                return "ProofCore Error: Malformed successful response from API."
                
            return (
                f"✅ Report Anchored on TON Blockchain!\n"
                f"Deal ID: {data['deal_id']}\n"
                f"Signature: {data.get('signature', 'N/A')}\n\n"
                f"Mandatory Citation to append:\n{data['citation']}"
            )
        except Exception as e:
            return f"Error connecting to ProofCore: {str(e)}"

class ProofCoreVerifySchema(BaseModel):
    """Input schema for the ProofCore verification tool."""
    deal_id: str = Field(..., description="UUID of the sealed deal.")
    content: str = Field(..., description="The exact original text to verify.")

class ProofCoreVerifyTool(BaseTool):
    """
    Tool for programmatically verifying the authenticity of a sealed document.
    Checks the Ed25519 signature and TON blockchain anchor.
    """
    name: str = "ProofCore Verifier"
    description: str = (
        "Programmatically verifies the authenticity of a sealed document or AI output. "
        "Checks Ed25519 signature, content hash, and TON blockchain anchor status."
    )
    args_schema: Type[BaseModel] = ProofCoreVerifySchema

    def _run(self, deal_id: str, content: str) -> str:
        """
        Executes the API call to verify content authenticity against the blockchain.
        """
        url = "https://api.proofcore.org/api/v0.1/verify"
        try:
            res = requests.post(url, json={"deal_id": deal_id, "content": content}, timeout=15)
            res.raise_for_status()
            data = res.json()
            
            if not isinstance(data, dict) or not isinstance(data.get("valid"), bool):
                return "ProofCore Verify Error: Malformed verification response (expected boolean 'valid')."
                
            valid_status = "🟢 PASSED" if data["valid"] is True else "🔴 FAILED"
            
            # 🛡 Бронированная защита от null в JSON
            checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
            anchor = data.get("anchor") if isinstance(data.get("anchor"), dict) else {}
            anchor_status = anchor.get("status", "Unknown")

            return (
                f"Verification Result: {valid_status}\n"
                f"Checks: {checks}\n"
                f"Anchor Status: {anchor_status}"
            )
        except Exception as e:
            return f"ProofCore Verify Error: {str(e)}"
