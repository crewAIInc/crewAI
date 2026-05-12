from unittest.mock import Mock, patch

from crewai_tools.tools.coinbase_agentic_wallet_tool import CoinbaseAgenticWalletTool


def test_coinbase_agentic_wallet_tool_defaults():
    tool = CoinbaseAgenticWalletTool()

    assert tool.command == "node"
    assert str(tool.bundle_path).endswith(".payments-mcp/bundle.js")
    assert tool.connect_timeout == 60
    assert tool.tool_names == ()


def test_coinbase_agentic_wallet_tool_starts_adapter_lazily():
    adapted_tools = ["search_bazaar", "pay"]
    adapter = Mock()
    adapter.tools = adapted_tools

    with (
        patch.object(CoinbaseAgenticWalletTool, "_server_params", return_value="params"),
        patch(
            "crewai_tools.tools.coinbase_agentic_wallet_tool.coinbase_agentic_wallet_tool.MCPServerAdapter",
            return_value=adapter,
        ) as adapter_class,
    ):
        tool = CoinbaseAgenticWalletTool("pay", connect_timeout=90)

        assert tool.tools == adapted_tools
        adapter_class.assert_called_once_with("params", "pay", connect_timeout=90)

        tool.stop()
        adapter.stop.assert_called_once()
