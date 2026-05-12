from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from crewai.tools import BaseTool

from crewai_tools.adapters.mcp_adapter import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection


class CoinbaseAgenticWalletTool:
    """Expose Coinbase Agentic Wallet MCP tools to CrewAI agents.

    Wraps the local MCP bundle installed by `npx @coinbase/payments-mcp` so
    agents can discover and pay for x402-monetized HTTP APIs with USDC through
    a Coinbase-managed embedded wallet.
    """

    command = "node"
    bundle_path: ClassVar[Path] = Path.home() / ".payments-mcp" / "bundle.js"

    def __init__(
        self,
        *tool_names: str,
        connect_timeout: int = 60,
    ) -> None:
        """Create a wrapper for the installed Coinbase Agentic Wallet MCP bundle."""
        self.tool_names = tool_names
        self.connect_timeout = connect_timeout
        self._adapter: MCPServerAdapter | None = None

    def _server_params(self) -> Any:
        """Build stdio parameters for the installed MCP bundle."""
        from mcp import StdioServerParameters

        return StdioServerParameters(
            command=self.command,
            args=[str(self.bundle_path)],
            env=None,
        )

    def start(self) -> ToolCollection[BaseTool]:
        """Start the MCP server and return the adapted CrewAI tools."""
        if self._adapter is None:
            if not self.bundle_path.is_file():
                raise FileNotFoundError(
                    "Coinbase Agentic Wallet MCP bundle not found at "
                    f"{self.bundle_path}. Run: "
                    "npx @coinbase/payments-mcp install --client other"
                )
            self._adapter = MCPServerAdapter(
                self._server_params(),
                *self.tool_names,
                connect_timeout=self.connect_timeout,
            )
        return self.tools

    @property
    def tools(self) -> ToolCollection[BaseTool]:
        """Return the adapted MCP tools, starting the server if needed."""
        if self._adapter is None:
            self.start()

        if self._adapter is None:
            raise RuntimeError("Coinbase Agentic Wallet MCP adapter did not start")

        return self._adapter.tools

    def stop(self) -> None:
        """Stop the MCP server if it has been started."""
        if self._adapter is not None:
            adapter = self._adapter
            self._adapter = None
            adapter.stop()

    def __enter__(self) -> ToolCollection[BaseTool]:
        """Start the MCP server and return the adapted CrewAI tools."""
        return self.start()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        """Stop the MCP server when leaving a context manager block."""
        self.stop()
