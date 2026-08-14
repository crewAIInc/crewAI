"""Mocked unit tests for USDCtoFiat CrewAI tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.usdctofiat_tool.usdctofiat_tool import (
    UsdctoFiatCashoutTool,
    UsdctoFiatDepositsTool,
    UsdctoFiatEstimateTool,
    UsdctoFiatWatchTool,
    UsdctoFiatWithdrawTool,
)


class _Prepared:
    def __init__(self, mode: str = "fast") -> None:
        self.mode = mode
        self.steps = (
            ["approve", "createDeposit"]
            if mode == "fast"
            else ["approve", "createDeposit", "setRateManager"]
        )
        self.attribution = {"referral_code": "TOFIAT", "referrers": ["galleonlabs"]}

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "steps": self.steps,
            "attribution": self.attribution,
        }


class _Cashout:
    def as_dict(self) -> dict:
        return {
            "deposit_id": "42",
            "tx_hash": "0x" + "ab" * 32,
            "mode": "fast",
        }


class _Estimate:
    def as_dict(self) -> dict:
        return {
            "mode": "fast",
            "amount_units": 100_000_000,
            "currency": "EUR",
            "spread_bps": 0,
            "manager_fee_bps": 0,
        }


class _Unsigned:
    def as_dict(self) -> dict:
        return {
            "to": "0x777777779d229cdF3110e9de47943791c26300Ef",
            "data": "0xwithdraw",
        }


class _ModeRequired(Exception):
    code = "VALIDATION"

    def __init__(self) -> None:
        super().__init__("mode is required: pass mode='fast' or mode='best'")


@pytest.fixture
def mock_offramp() -> MagicMock:
    client = MagicMock()
    client.prepare.return_value = _Prepared("fast")
    client.cashout.return_value = _Cashout()
    client.estimate.return_value = _Estimate()
    client.deposits.return_value = [{"id": "42", "status": "ACTIVE"}]
    client.watch.return_value = iter([{"id": "42", "status": "ACTIVE"}])
    client.withdraw.return_value = _Unsigned()
    return client


@pytest.fixture
def tools(mock_offramp: MagicMock):
    with patch(
        "crewai_tools.tools.usdctofiat_tool.usdctofiat_tool._create_offramp",
        return_value=mock_offramp,
    ):
        yield {
            "cashout": UsdctoFiatCashoutTool(),
            "estimate": UsdctoFiatEstimateTool(),
            "watch": UsdctoFiatWatchTool(),
            "withdraw": UsdctoFiatWithdrawTool(),
            "deposits": UsdctoFiatDepositsTool(),
        }, mock_offramp


def test_docstring_discloses_product_and_docs() -> None:
    import crewai_tools.tools.usdctofiat_tool.usdctofiat_tool as module

    text = f"{UsdctoFiatCashoutTool.__doc__ or ''} {module.__doc__ or ''}".lower()
    assert "usdctofiat" in text
    assert "galleon" in text
    assert "usdctofiat.xyz/developers" in text


def test_mode_is_not_a_constructor_default() -> None:
    with pytest.raises(TypeError, match="does not default mode"):
        UsdctoFiatCashoutTool(mode="fast")
    with pytest.raises(TypeError, match="does not default mode"):
        UsdctoFiatEstimateTool(mode="best")
    UsdctoFiatCashoutTool()


def test_no_private_key_constructor() -> None:
    with pytest.raises(TypeError, match="does not accept a private key"):
        UsdctoFiatCashoutTool(private_key="0xabc")
    with pytest.raises(TypeError, match="does not accept a private key"):
        UsdctoFiatCashoutTool(evm_private_key="0xabc")
    kit = UsdctoFiatCashoutTool()
    assert not hasattr(kit, "private_key")
    assert kit.signer is None


def test_args_schema_requires_mode() -> None:
    fields = UsdctoFiatCashoutTool.args_schema.model_fields
    assert "mode" in fields
    assert fields["mode"].is_required()


def test_cashout_without_signer_returns_unsigned_prepare(tools) -> None:
    kit, offramp = tools
    payload = json.loads(
        kit["cashout"]._run(
            mode="fast", amount="100", currency="EUR", platform="revolut", payee="alice"
        )
    )
    assert payload["signed"] is False
    assert payload["prepared"]["mode"] == "fast"
    assert payload["prepared"]["steps"] == ["approve", "createDeposit"]
    assert payload["prepared"]["attribution"]["referral_code"] == "TOFIAT"
    offramp.prepare.assert_called_once()
    offramp.cashout.assert_not_called()


def test_cashout_with_injected_signer(mock_offramp: MagicMock) -> None:
    def signer(tx):
        return {"hash": "0x" + "cd" * 32, "deposit_id": "42"}

    with patch(
        "crewai_tools.tools.usdctofiat_tool.usdctofiat_tool._create_offramp",
        return_value=mock_offramp,
    ):
        kit = UsdctoFiatCashoutTool(signer=signer)
        payload = json.loads(
            kit._run(mode="fast", amount="10", currency="GBP", platform="monzo", payee="alice")
        )
    assert payload["signed"] is True
    assert payload["result"]["deposit_id"] == "42"
    assert payload["result"]["mode"] == "fast"
    mock_offramp.cashout.assert_called_once()
    kwargs = mock_offramp.cashout.call_args.kwargs
    assert kwargs["mode"] == "fast"
    assert kwargs["signer"] is signer


def test_cashout_mode_required_is_returned_as_json(tools) -> None:
    kit, offramp = tools
    offramp.prepare.side_effect = _ModeRequired()
    payload = json.loads(
        kit["cashout"]._run(
            mode="", amount="100", currency="EUR", platform="revolut", payee="alice"
        )
    )
    assert "mode is required" in payload["error"]
    assert payload["code"] == "VALIDATION"


def test_estimate_watch_withdraw_deposits(tools) -> None:
    kit, _offramp = tools
    estimate = json.loads(kit["estimate"]._run(mode="fast", amount="100", currency="EUR"))
    assert estimate["spread_bps"] == 0
    assert estimate["manager_fee_bps"] == 0
    assert estimate["mode"] == "fast"

    watched = json.loads(kit["watch"]._run("42"))
    assert watched["snapshots"][0]["status"] == "ACTIVE"

    rows = json.loads(kit["deposits"]._run("0x1111111111111111111111111111111111111111"))
    assert rows["deposits"][0]["id"] == "42"

    withdrawn = json.loads(kit["withdraw"]._run("42"))
    assert withdrawn["to"].lower().endswith("ef")
    closed = json.loads(kit["withdraw"].close("42"))
    assert closed["data"] == "0xwithdraw"


def test_estimate_mode_required(tools) -> None:
    kit, offramp = tools
    offramp.estimate.side_effect = _ModeRequired()
    payload = json.loads(kit["estimate"]._run(mode="slow", amount="100", currency="EUR"))
    assert "mode is required" in payload["error"]
