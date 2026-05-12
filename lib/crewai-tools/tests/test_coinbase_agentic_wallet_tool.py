from unittest.mock import Mock, patch

import pytest

from crewai_tools.tools.coinbase_agentic_wallet_tool import CoinbaseAgenticWalletTool


def test_coinbase_agentic_wallet_tool_defaults():
    tool = CoinbaseAgenticWalletTool()

    assert tool.command == "node"
    assert str(tool.bundle_path).endswith(".payments-mcp/bundle.js")
    assert tool.connect_timeout == 60
    assert tool.tool_names == ()


def test_coinbase_agentic_wallet_tool_starts_adapter_lazily(tmp_path):
    adapted_tools = ["search_bazaar", "pay"]
    bundle_path = tmp_path / "bundle.js"
    bundle_path.write_text("")
    adapter = Mock()
    adapter.tools = adapted_tools

    with (
        patch.object(CoinbaseAgenticWalletTool, "bundle_path", bundle_path),
        patch.object(CoinbaseAgenticWalletTool, "_server_params", return_value="params"),
        patch(
            "crewai_tools.tools.coinbase_agentic_wallet_tool.coinbase_agentic_wallet_tool.MCPServerAdapter",
            return_value=adapter,
        ) as adapter_class,
    ):
        tool = CoinbaseAgenticWalletTool("pay", connect_timeout=90)

        adapter_class.assert_not_called()
        assert tool._adapter is None
        assert tool.tools == adapted_tools
        adapter_class.assert_called_once_with("params", "pay", connect_timeout=90)

        tool.stop()
        adapter.stop.assert_called_once()


def test_coinbase_agentic_wallet_tool_requires_installed_bundle(tmp_path):
    missing_bundle = tmp_path / "missing-bundle.js"

    with (
        patch.object(CoinbaseAgenticWalletTool, "bundle_path", missing_bundle),
        patch(
            "crewai_tools.tools.coinbase_agentic_wallet_tool.coinbase_agentic_wallet_tool.MCPServerAdapter",
        ) as adapter_class,
    ):
        tool = CoinbaseAgenticWalletTool()

        with pytest.raises(
            FileNotFoundError,
            match="npx @coinbase/payments-mcp install --client other",
        ):
            tool.start()
        adapter_class.assert_not_called()
