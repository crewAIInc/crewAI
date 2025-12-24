"""Kraken Futures Account Tools - Private endpoints voor account informatie."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


def _balance_threshold() -> float:
    raw = os.getenv("KRAKEN_BALANCE_MIN_THRESHOLD", "0.000001").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.000001
    return max(value, 0.0)


def _has_significant_value(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_significant_value(v, threshold) for v in value.values())
    if isinstance(value, list):
        return any(_has_significant_value(v, threshold) for v in value)
    if isinstance(value, bool):
        return bool(value)
    try:
        return abs(float(value)) >= threshold
    except (TypeError, ValueError):
        return True


def _filter_currency_map(payload: Any, threshold: float) -> Any:
    if not isinstance(payload, dict):
        return payload
    return {
        key: value
        for key, value in payload.items()
        if _has_significant_value(value, threshold)
    }


def _filter_wallets(payload: dict[str, Any], threshold: float) -> dict[str, Any]:
    accounts = payload.get("accounts")
    if not isinstance(accounts, dict):
        return payload
    new_accounts: dict[str, Any] = {}
    for name, account in accounts.items():
        if not isinstance(account, dict):
            continue
        filtered: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        has_signal = False
        for key, value in account.items():
            if key in ("balances", "currencies"):
                filtered_map = _filter_currency_map(value, threshold)
                if filtered_map:
                    filtered[key] = filtered_map
                    has_signal = True
                continue
            if key in ("currency", "type"):
                metadata[key] = value
                continue
            if _has_significant_value(value, threshold):
                filtered[key] = value
                has_signal = True
        if has_signal:
            filtered.update(metadata)
            new_accounts[name] = filtered
    new_payload = dict(payload)
    new_payload["accounts"] = new_accounts
    return new_payload


# =============================================================================
# Tool 1: Haal Wallets Op
# =============================================================================
class KrakenFuturesGetWalletsTool(KrakenFuturesBaseTool):
    """Haal wallet informatie op."""

    name: str = "kraken_futures_get_wallets"
    description: str = "Haal wallet saldo's op inclusief beschikbare margin, equity, portfolio waarde en collateral per asset."

    def _run(self) -> str:
        """Haal wallets op van Kraken Futures."""
        result = self._private_request("accounts", method="GET")
        return str(_filter_wallets(result, _balance_threshold()))


# =============================================================================
# Tool 2: Haal Open Posities Op
# =============================================================================
class KrakenFuturesGetOpenPositionsTool(KrakenFuturesBaseTool):
    """Haal open posities op."""

    name: str = "kraken_futures_get_open_positions"
    description: str = "Haal alle open Futures posities op met details zoals grootte, entry prijs, ongerealiseerde PnL en liquidatie prijs."

    def _run(self) -> str:
        """Haal open posities op van Kraken Futures."""
        result = self._private_request("openpositions", method="GET")
        return str(result)


# =============================================================================
# Tool 3: Haal Portfolio Margin Parameters Op
# =============================================================================
class KrakenFuturesGetPortfolioMarginInput(BaseModel):
    """Input schema voor KrakenFuturesGetPortfolioMarginTool."""

    portfolio: str | None = Field(
        default=None, description="Portfolio naam (optioneel, standaard is 'main')"
    )


class KrakenFuturesGetPortfolioMarginTool(KrakenFuturesBaseTool):
    """Haal portfolio margin parameters op."""

    name: str = "kraken_futures_get_portfolio_margin"
    description: str = "Haal portfolio margin parameters op inclusief maintenance margin, initial margin en beschikbare margin."
    args_schema: type[BaseModel] = KrakenFuturesGetPortfolioMarginInput

    def _run(self, portfolio: str | None = None) -> str:
        """Haal portfolio margin op van Kraken Futures."""
        data: dict[str, Any] = {}
        if portfolio:
            data["portfolio"] = portfolio
        result = self._private_request("initialmargin", data, method="GET")
        return str(result)


# =============================================================================
# Tool 4: Bereken Portfolio Margin PnL
# =============================================================================
class KrakenFuturesCalculateMarginPnLInput(BaseModel):
    """Input schema voor KrakenFuturesCalculateMarginPnLTool."""

    portfolio: str | None = Field(
        default=None, description="Portfolio naam"
    )


class KrakenFuturesCalculateMarginPnLTool(KrakenFuturesBaseTool):
    """Bereken portfolio margin PnL en Greeks."""

    name: str = "kraken_futures_calculate_margin_pnl"
    description: str = "Bereken portfolio margin PnL en Greeks inclusief delta, gamma en scenario analyses."
    args_schema: type[BaseModel] = KrakenFuturesCalculateMarginPnLInput

    def _run(self, portfolio: str | None = None) -> str:
        """Bereken margin PnL op Kraken Futures."""
        data: dict[str, Any] = {}
        if portfolio:
            data["portfolio"] = portfolio
        result = self._private_request("pnlcurrency", data, method="GET")
        return str(result)


# =============================================================================
# Tool 5: Haal Unwind Queue Positie Op
# =============================================================================
class KrakenFuturesGetUnwindQueueTool(KrakenFuturesBaseTool):
    """Haal positie percentiel in unwind queue op."""

    name: str = "kraken_futures_get_unwind_queue"
    description: str = "Haal positie percentiel op in de unwind wachtrij. Lager percentiel betekent eerder gelikwideerd worden bij deleveraging."

    def _run(self) -> str:
        """Haal unwind queue positie op van Kraken Futures."""
        result = self._private_request("unwindqueue", method="GET")
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetWalletsTool",
    "KrakenFuturesGetOpenPositionsTool",
    "KrakenFuturesGetPortfolioMarginTool",
    "KrakenFuturesCalculateMarginPnLTool",
    "KrakenFuturesGetUnwindQueueTool",
]
