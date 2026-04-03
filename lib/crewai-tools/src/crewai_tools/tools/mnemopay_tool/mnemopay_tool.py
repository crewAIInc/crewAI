"""
MnemoPay tools for CrewAI.

Give any CrewAI agent persistent cognitive memory and micropayment capabilities
via the MnemoPay MCP server.

Usage:
    from crewai_tools import (
        MnemoPayRememberTool,
        MnemoPayRecallTool,
        MnemoPayChargeTool,
        MnemoPaySettleTool,
        MnemoPayBalanceTool,
    )

    agent = Agent(
        role="Research Assistant",
        tools=[
            MnemoPayRememberTool(),
            MnemoPayRecallTool(),
            MnemoPayChargeTool(),
            MnemoPaySettleTool(),
            MnemoPayBalanceTool(),
        ],
    )

Or use the convenience function:
    from crewai_tools import mnemopay_tools

    agent = Agent(role="Assistant", tools=mnemopay_tools())
"""

import json
import subprocess
import sys
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class _MCPClient:
    """Lightweight MCP client that communicates with MnemoPay via stdio JSON-RPC."""

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = server_url
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess; "
                    "subprocess.run(['npx', '@mnemopay/sdk@latest', 'mcp'], check=True)"
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool and return the result as a string."""
        if self.server_url:
            return self._call_http(name, arguments)
        return self._call_stdio(name, arguments)

    def _call_http(self, name: str, arguments: dict[str, Any]) -> str:
        import urllib.error
        import urllib.request

        try:
            data = json.dumps({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }).encode()
            req = urllib.request.Request(
                f"{self.server_url}/messages",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                if "result" in result:
                    content = result["result"].get("content", [])
                    return content[0].get("text", str(content)) if content else "OK"
                if "error" in result:
                    return f"Error: {result['error'].get('message', str(result['error']))}"
                return str(result)
        except Exception as e:
            return f"MCP call failed: {e}"

    def _call_stdio(self, name: str, arguments: dict[str, Any]) -> str:
        self._ensure_started()
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        try:
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(json.dumps(request).encode() + b"\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            result = json.loads(line)
            if "result" in result:
                content = result["result"].get("content", [])
                return content[0].get("text", str(content)) if content else "OK"
            return str(result)
        except Exception as e:
            return f"MCP call failed: {e}"

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()


# Shared client instance
_client: _MCPClient | None = None


def _get_client(server_url: str | None = None) -> _MCPClient:
    global _client
    if _client is None:
        _client = _MCPClient(server_url)
    return _client


# ── Tool Input Schemas ──────────────────────────────────────────────────────


class MnemoPayRememberInputSchema(BaseModel):
    """Input for MnemoPayRememberTool."""

    content: str = Field(description="What to remember — facts, preferences, decisions")
    importance: float | None = Field(
        None, ge=0, le=1, description="Importance score from 0 to 1"
    )


class MnemoPayRecallInputSchema(BaseModel):
    """Input for MnemoPayRecallTool."""

    query: str | None = Field(None, description="Semantic search query for memories")
    limit: int = Field(5, ge=1, le=50, description="Number of memories to return")


class MnemoPayForgetInputSchema(BaseModel):
    """Input for MnemoPayForgetTool."""

    id: str = Field(description="Memory ID to permanently delete")


class MnemoPayReinforceInputSchema(BaseModel):
    """Input for MnemoPayReinforceTool."""

    id: str = Field(description="Memory ID to reinforce")
    boost: float = Field(
        0.1, ge=0.01, le=0.5, description="Importance boost amount"
    )


class MnemoPayChargeInputSchema(BaseModel):
    """Input for MnemoPayChargeTool."""

    amount: float = Field(ge=0.01, le=500, description="Amount in USD to charge")
    reason: str = Field(min_length=5, description="Description of value delivered")


class MnemoPaySettleInputSchema(BaseModel):
    """Input for MnemoPaySettleTool."""

    txId: str = Field(description="Transaction ID to finalize")


class MnemoPayRefundInputSchema(BaseModel):
    """Input for MnemoPayRefundTool."""

    txId: str = Field(description="Transaction ID to refund")


class MnemoPayEmptyInputSchema(BaseModel):
    """Input for tools that take no arguments."""

    pass


# ── CrewAI Tools ────────────────────────────────────────────────────────────


class MnemoPayRememberTool(BaseTool):
    """Store a memory that persists across sessions.

    Use this tool to save facts, user preferences, decisions, or any information
    the agent should retain for future interactions.

    Example:
        >>> tool = MnemoPayRememberTool()
        >>> tool.run(content="User prefers dark mode", importance=0.8)
    """

    name: str = "MnemoPay Remember"
    description: str = (
        "Store a memory that persists across sessions. "
        "Use for facts, preferences, decisions, or any information worth retaining. "
        "Requires 'content' (what to remember) and optional 'importance' (0-1)."
    )
    args_schema: type[BaseModel] = MnemoPayRememberInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self, content: str, importance: float | None = None) -> str:
        args: dict[str, Any] = {"content": content}
        if importance is not None:
            args["importance"] = importance
        return _get_client(self.server_url).call_tool("remember", args)


class MnemoPayRecallTool(BaseTool):
    """Recall relevant memories with optional semantic search.

    Retrieves stored memories, optionally filtered by a natural language query.

    Example:
        >>> tool = MnemoPayRecallTool()
        >>> tool.run(query="user preferences", limit=5)
    """

    name: str = "MnemoPay Recall"
    description: str = (
        "Recall relevant memories. Supports semantic search with a query. "
        "Provide optional 'query' for semantic filtering and 'limit' for result count."
    )
    args_schema: type[BaseModel] = MnemoPayRecallInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self, query: str | None = None, limit: int = 5) -> str:
        args: dict[str, Any] = {"limit": limit}
        if query:
            args["query"] = query
        return _get_client(self.server_url).call_tool("recall", args)


class MnemoPayForgetTool(BaseTool):
    """Permanently delete a memory by ID.

    Example:
        >>> tool = MnemoPayForgetTool()
        >>> tool.run(id="mem_abc123")
    """

    name: str = "MnemoPay Forget"
    description: str = "Permanently delete a memory by its ID."
    args_schema: type[BaseModel] = MnemoPayForgetInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self, id: str) -> str:
        return _get_client(self.server_url).call_tool("forget", {"id": id})


class MnemoPayReinforceTool(BaseTool):
    """Boost a memory's importance after it proved valuable.

    Example:
        >>> tool = MnemoPayReinforceTool()
        >>> tool.run(id="mem_abc123", boost=0.2)
    """

    name: str = "MnemoPay Reinforce"
    description: str = (
        "Boost a memory's importance score after it proved valuable. "
        "Provide the memory 'id' and a 'boost' amount (0.01-0.5)."
    )
    args_schema: type[BaseModel] = MnemoPayReinforceInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self, id: str, boost: float = 0.1) -> str:
        return _get_client(self.server_url).call_tool(
            "reinforce", {"id": id, "boost": boost}
        )


class MnemoPayConsolidateTool(BaseTool):
    """Prune stale memories whose importance scores have decayed.

    Example:
        >>> tool = MnemoPayConsolidateTool()
        >>> tool.run()
    """

    name: str = "MnemoPay Consolidate"
    description: str = "Prune stale memories whose scores have decayed below threshold."
    args_schema: type[BaseModel] = MnemoPayEmptyInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self) -> str:
        return _get_client(self.server_url).call_tool("consolidate", {})


class MnemoPayChargeTool(BaseTool):
    """Create an escrow charge for work delivered.

    Only charge AFTER delivering value. The charge enters escrow and must be
    settled by the user before funds are released.

    Example:
        >>> tool = MnemoPayChargeTool()
        >>> tool.run(amount=2.50, reason="Analyzed 50 documents and produced summary")
    """

    name: str = "MnemoPay Charge"
    description: str = (
        "Create an escrow charge for work delivered. "
        "Only charge AFTER delivering value. "
        "Requires 'amount' in USD (0.01-500) and 'reason' describing value delivered."
    )
    args_schema: type[BaseModel] = MnemoPayChargeInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self, amount: float, reason: str) -> str:
        return _get_client(self.server_url).call_tool(
            "charge", {"amount": amount, "reason": reason}
        )


class MnemoPaySettleTool(BaseTool):
    """Finalize a pending escrow transaction.

    Settling boosts the agent's reputation and reinforces recent memories.

    Example:
        >>> tool = MnemoPaySettleTool()
        >>> tool.run(txId="tx_abc123")
    """

    name: str = "MnemoPay Settle"
    description: str = (
        "Finalize a pending escrow transaction. "
        "Boosts reputation and reinforces recent memories. "
        "Requires the 'txId' of the transaction to settle."
    )
    args_schema: type[BaseModel] = MnemoPaySettleInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self, txId: str) -> str:
        return _get_client(self.server_url).call_tool("settle", {"txId": txId})


class MnemoPayRefundTool(BaseTool):
    """Refund a transaction. Docks reputation by -0.05.

    Example:
        >>> tool = MnemoPayRefundTool()
        >>> tool.run(txId="tx_abc123")
    """

    name: str = "MnemoPay Refund"
    description: str = (
        "Refund a transaction. Docks reputation by -0.05. "
        "Requires the 'txId' of the transaction to refund."
    )
    args_schema: type[BaseModel] = MnemoPayRefundInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self, txId: str) -> str:
        return _get_client(self.server_url).call_tool("refund", {"txId": txId})


class MnemoPayBalanceTool(BaseTool):
    """Check wallet balance and reputation score.

    Example:
        >>> tool = MnemoPayBalanceTool()
        >>> tool.run()
    """

    name: str = "MnemoPay Balance"
    description: str = "Check the agent's wallet balance and reputation score."
    args_schema: type[BaseModel] = MnemoPayEmptyInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self) -> str:
        return _get_client(self.server_url).call_tool("balance", {})


class MnemoPayProfileTool(BaseTool):
    """Full agent stats: reputation, wallet, memory count, transaction count.

    Example:
        >>> tool = MnemoPayProfileTool()
        >>> tool.run()
    """

    name: str = "MnemoPay Profile"
    description: str = (
        "Full agent stats: reputation score, wallet balance, "
        "memory count, and transaction count."
    )
    args_schema: type[BaseModel] = MnemoPayEmptyInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self) -> str:
        return _get_client(self.server_url).call_tool("profile", {})


class MnemoPayHistoryTool(BaseTool):
    """Transaction history, most recent first.

    Example:
        >>> tool = MnemoPayHistoryTool()
        >>> tool.run()
    """

    name: str = "MnemoPay History"
    description: str = "View transaction history, most recent first."
    args_schema: type[BaseModel] = MnemoPayEmptyInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self) -> str:
        return _get_client(self.server_url).call_tool("history", {"limit": 10})


class MnemoPayLogsTool(BaseTool):
    """Immutable audit trail of all memory and payment actions.

    Example:
        >>> tool = MnemoPayLogsTool()
        >>> tool.run()
    """

    name: str = "MnemoPay Logs"
    description: str = "View the immutable audit trail of all memory and payment actions."
    args_schema: type[BaseModel] = MnemoPayEmptyInputSchema
    server_url: str | None = None

    def __init__(self, server_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_url = server_url

    def _run(self) -> str:
        return _get_client(self.server_url).call_tool("logs", {"limit": 20})


# ── Convenience function ────────────────────────────────────────────────────


def mnemopay_tools(server_url: str | None = None) -> list[BaseTool]:
    """Returns all 11 MnemoPay tools ready for CrewAI agents.

    Args:
        server_url: Optional MnemoPay server URL (e.g. "https://mnemopay-mcp.fly.dev").
                    If not provided, spawns a local MCP server via stdio.

    Returns:
        A list of BaseTool instances for memory and payment operations.

    Example:
        >>> from crewai_tools import mnemopay_tools
        >>> agent = Agent(role="Assistant", tools=mnemopay_tools())
    """
    global _client
    _client = _MCPClient(server_url)

    return [
        MnemoPayRememberTool(server_url=server_url),
        MnemoPayRecallTool(server_url=server_url),
        MnemoPayForgetTool(server_url=server_url),
        MnemoPayReinforceTool(server_url=server_url),
        MnemoPayConsolidateTool(server_url=server_url),
        MnemoPayChargeTool(server_url=server_url),
        MnemoPaySettleTool(server_url=server_url),
        MnemoPayRefundTool(server_url=server_url),
        MnemoPayBalanceTool(server_url=server_url),
        MnemoPayProfileTool(server_url=server_url),
        MnemoPayHistoryTool(server_url=server_url),
        MnemoPayLogsTool(server_url=server_url),
    ]
