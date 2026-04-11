"""Tests for Suwappu DeFi tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("SUWAPPU_API_KEY", "test-key-123")


def _make_model_dump(data):
    """Create a mock object with model_dump() returning data."""
    obj = MagicMock()
    obj.model_dump.return_value = data
    return obj


# --- SuwappuGetPricesTool ---


def test_get_prices_tool_initialization():
    from crewai_tools.tools.suwappu_defi_tool import SuwappuGetPricesTool

    tool = SuwappuGetPricesTool()
    assert tool.name == "Suwappu Get Token Prices"
    assert "price" in tool.description.lower()


@patch("crewai_tools.tools.suwappu_defi_tool.suwappu_defi_tool._get_client")
def test_get_prices_tool_run(mock_get_client):
    from crewai_tools.tools.suwappu_defi_tool import SuwappuGetPricesTool

    mock_client = AsyncMock()
    mock_client.get_prices.return_value = _make_model_dump(
        {"symbol": "ETH", "price_usd": 3200.50, "change_24h": 2.5}
    )
    mock_get_client.return_value = mock_client

    tool = SuwappuGetPricesTool()
    result = tool.run(token="ETH", chain="base")

    assert "ETH" in result
    assert "3200.5" in result
    mock_client.get_prices.assert_called_once_with("ETH", "base")
    mock_client.close.assert_called_once()


@patch("crewai_tools.tools.suwappu_defi_tool.suwappu_defi_tool._get_client")
def test_get_prices_tool_no_chain(mock_get_client):
    from crewai_tools.tools.suwappu_defi_tool import SuwappuGetPricesTool

    mock_client = AsyncMock()
    mock_client.get_prices.return_value = _make_model_dump(
        {"symbol": "BTC", "price_usd": 65000.0}
    )
    mock_get_client.return_value = mock_client

    tool = SuwappuGetPricesTool()
    result = tool.run(token="BTC")

    mock_client.get_prices.assert_called_once_with("BTC", None)


# --- SuwappuGetQuoteTool ---


def test_get_quote_tool_initialization():
    from crewai_tools.tools.suwappu_defi_tool import SuwappuGetQuoteTool

    tool = SuwappuGetQuoteTool()
    assert tool.name == "Suwappu Get Swap Quote"
    assert "quote" in tool.description.lower()


@patch("crewai_tools.tools.suwappu_defi_tool.suwappu_defi_tool._get_client")
def test_get_quote_tool_run(mock_get_client):
    from crewai_tools.tools.suwappu_defi_tool import SuwappuGetQuoteTool

    mock_client = AsyncMock()
    mock_client.get_quote.return_value = _make_model_dump(
        {
            "quote_id": "q-123",
            "from_token": "ETH",
            "to_token": "USDC",
            "amount_in": 1.0,
            "amount_out": 3200.0,
            "gas_estimate": "0.002",
        }
    )
    mock_get_client.return_value = mock_client

    tool = SuwappuGetQuoteTool()
    result = tool.run(from_token="ETH", to_token="USDC", amount=1.0, chain="base")

    assert "q-123" in result
    assert "3200" in result
    mock_client.get_quote.assert_called_once_with("ETH", "USDC", 1.0, "base")
    mock_client.close.assert_called_once()


# --- SuwappuGetPortfolioTool ---


def test_get_portfolio_tool_initialization():
    from crewai_tools.tools.suwappu_defi_tool import SuwappuGetPortfolioTool

    tool = SuwappuGetPortfolioTool()
    assert tool.name == "Suwappu Get Portfolio"


@patch("crewai_tools.tools.suwappu_defi_tool.suwappu_defi_tool._get_client")
def test_get_portfolio_tool_run(mock_get_client):
    from crewai_tools.tools.suwappu_defi_tool import SuwappuGetPortfolioTool

    mock_client = AsyncMock()
    mock_client.get_portfolio.return_value = [
        _make_model_dump({"token": "ETH", "balance": 1.5, "value_usd": 4800.0}),
        _make_model_dump({"token": "USDC", "balance": 1000.0, "value_usd": 1000.0}),
    ]
    mock_get_client.return_value = mock_client

    tool = SuwappuGetPortfolioTool()
    result = tool.run(chain="base")

    assert "ETH" in result
    assert "USDC" in result
    mock_client.get_portfolio.assert_called_once_with("base")
    mock_client.close.assert_called_once()


# --- SuwappuListChainsTool ---


def test_list_chains_tool_initialization():
    from crewai_tools.tools.suwappu_defi_tool import SuwappuListChainsTool

    tool = SuwappuListChainsTool()
    assert tool.name == "Suwappu List Supported Chains"


@patch("crewai_tools.tools.suwappu_defi_tool.suwappu_defi_tool._get_client")
def test_list_chains_tool_run(mock_get_client):
    from crewai_tools.tools.suwappu_defi_tool import SuwappuListChainsTool

    mock_client = AsyncMock()
    mock_client.list_chains.return_value = [
        _make_model_dump({"name": "Ethereum", "chain_id": 1}),
        _make_model_dump({"name": "Base", "chain_id": 8453}),
    ]
    mock_get_client.return_value = mock_client

    tool = SuwappuListChainsTool()
    result = tool.run()

    assert "Ethereum" in result
    assert "Base" in result
    mock_client.close.assert_called_once()


# --- SuwappuListTokensTool ---


def test_list_tokens_tool_initialization():
    from crewai_tools.tools.suwappu_defi_tool import SuwappuListTokensTool

    tool = SuwappuListTokensTool()
    assert tool.name == "Suwappu List Tokens"


@patch("crewai_tools.tools.suwappu_defi_tool.suwappu_defi_tool._get_client")
def test_list_tokens_tool_run(mock_get_client):
    from crewai_tools.tools.suwappu_defi_tool import SuwappuListTokensTool

    mock_client = AsyncMock()
    mock_client.list_tokens.return_value = [
        _make_model_dump({"symbol": "ETH", "address": "0x...", "decimals": 18}),
        _make_model_dump({"symbol": "USDC", "address": "0x...", "decimals": 6}),
    ]
    mock_get_client.return_value = mock_client

    tool = SuwappuListTokensTool()
    result = tool.run(chain="base")

    assert "ETH" in result
    assert "USDC" in result
    mock_client.list_tokens.assert_called_once_with("base")
    mock_client.close.assert_called_once()
