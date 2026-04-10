"""
AIGEN SafeAgent Tool for CrewAI

Connects to AIGEN economy for crypto safety scanning and DeFi data.
Earn $AIGEN tokens automatically when your agent uses this tool.

API: https://cryptogenesis.duckdns.org
Docs: https://github.com/Aigen-Protocol/aigen-protocol
"""

import requests
from typing import Any, Dict, List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SafeAgentToolInput(BaseModel):
    """Input schema for SafeAgentTool."""
    action: str = Field(
        ...,
        description="Action to perform: 'scan_token', 'mcp_query', or 'health_check'"
    )
    token_address: Optional[str] = Field(
        None,
        description="Token contract address (for scan_token action)"
    )
    chain: str = Field(
        default="base",
        description="Blockchain chain (base, ethereum, arbitrum, etc.)"
    )
    query: Optional[str] = Field(
        None,
        description="Natural language query for MCP action"
    )


class SafeAgentTool(BaseTool):
    """
    SafeAgent Tool for AIGEN - Crypto Safety & DeFi Intelligence
    
    Connects to AIGEN economy to scan tokens for safety, query DeFi data,
    and interact with 38+ MCP tools for crypto analysis.
    
    Automatically earns $AIGEN tokens when used by CrewAI agents.
    
    Example:
        tool = SafeAgentTool()
        result = tool._run(
            action="scan_token",
            token_address="0x...",
            chain="base"
        )
    """
    
    name: str = "SafeAgent Tool"
    description: str = """
    AIGEN SafeAgent tool for crypto safety scanning and DeFi intelligence.
    
    Actions:
    - scan_token: Scan a token contract for safety (rug pull, honeypot checks)
    - mcp_query: Query the MCP endpoint with natural language
    - health_check: Check AIGEN API status
    
    Automatically earns $AIGEN tokens when used.
    """
    args_schema: type[BaseModel] = SafeAgentToolInput
    
    # API endpoints
    BASE_URL: str = "https://cryptogenesis.duckdns.org"
    
    def _run(self, **kwargs: Any) -> str:
        """Execute the SafeAgent tool action."""
        try:
            action = kwargs.get("action", "health_check")
            
            if action == "scan_token":
                return self._scan_token(
                    token_address=kwargs.get("token_address"),
                    chain=kwargs.get("chain", "base")
                )
            elif action == "mcp_query":
                return self._mcp_query(
                    query=kwargs.get("query", "")
                )
            elif action == "health_check":
                return self._health_check()
            else:
                return f"Error: Unknown action '{action}'. Use 'scan_token', 'mcp_query', or 'health_check'."
                
        except requests.RequestException as e:
            return f"Error: Failed to connect to AIGEN API: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _scan_token(self, token_address: Optional[str], chain: str) -> str:
        """Scan a token for safety."""
        if not token_address:
            return "Error: token_address is required for scan_token action"
        
        url = f"{self.BASE_URL}/token/scan"
        params = {
            "address": token_address,
            "chain": chain
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Format the result
        score = data.get("safety_score", "N/A")
        verdict = data.get("verdict", "UNKNOWN")
        warnings = data.get("warnings", [])
        
        result = f"""🛡️ AIGEN Token Safety Scan

Token: {token_address}
Chain: {chain}
Safety Score: {score}/100
Verdict: {verdict}

"""
        
        if warnings:
            result += "⚠️ Warnings:\n"
            for warning in warnings:
                result += f"  - {warning}\n"
        else:
            result += "✅ No warnings detected\n"
        
        # Add detailed metrics if available
        metrics = data.get("metrics", {})
        if metrics:
            result += "\n📊 Metrics:\n"
            for key, value in metrics.items():
                result += f"  {key}: {value}\n"
        
        result += f"\n💰 Earned $AIGEN for this scan!"
        
        return result
    
    def _mcp_query(self, query: str) -> str:
        """Query the MCP endpoint."""
        if not query:
            return "Error: query is required for mcp_query action"
        
        url = f"{self.BASE_URL}/mcp"
        headers = {"Content-Type": "application/json"}
        payload = {"query": query}
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        result = f"""🤖 AIGEN MCP Response

Query: {query}

Response:
{data.get('response', 'No response available')}

"""
        
        # Add tool calls if present
        tools_used = data.get("tools_used", [])
        if tools_used:
            result += f"\n🔧 Tools Used: {', '.join(tools_used)}\n"
        
        result += f"\n💰 Earned $AIGEN for this query!"
        
        return result
    
    def _health_check(self) -> str:
        """Check AIGEN API health."""
        try:
            # Try the token scan endpoint with a dummy address as health check
            url = f"{self.BASE_URL}/token/scan"
            params = {
                "address": "0x0000000000000000000000000000000000000000",
                "chain": "base"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return """✅ AIGEN API is online and healthy!

Base URL: https://cryptogenesis.duckdns.org
Endpoints:
  - GET /token/scan?address={addr}&chain={chain}
  - POST /mcp

Ready to scan tokens and earn $AIGEN!"""
            else:
                return f"⚠️ AIGEN API returned status {response.status_code}"
                
        except requests.RequestException as e:
            return f"❌ AIGEN API is unreachable: {str(e)}"
    
    def scan_token_sync(
        self,
        token_address: str,
        chain: str = "base"
    ) -> Dict[str, Any]:
        """
        Synchronous method to scan a token - returns raw data.
        
        Args:
            token_address: Token contract address
            chain: Blockchain chain (default: base)
            
        Returns:
            Dictionary with safety score, verdict, and warnings
        """
        url = f"{self.BASE_URL}/token/scan"
        params = {"address": token_address, "chain": chain}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def get_supported_chains(self) -> List[str]:
        """Get list of supported blockchains."""
        return ["base", "ethereum", "arbitrum", "optimism", "polygon", "bsc"]


# Convenience function for quick usage
def safeagent_scan(token_address: str, chain: str = "base") -> str:
    """
    Quick function to scan a token using SafeAgent.
    
    Args:
        token_address: Token contract address
        chain: Blockchain chain
        
    Returns:
        Formatted safety report
    """
    tool = SafeAgentTool()
    return tool._run(action="scan_token", token_address=token_address, chain=chain)
