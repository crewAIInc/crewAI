"""MadeOnSol tools — Solana & Robinhood Chain trading intelligence.

Wraps the `madeonsol-x402` SDK (https://pypi.org/project/madeonsol-x402/) to
give agents access to MadeOnSol's KOL wallet tracking (2,000+ tracked wallets),
Pump.fun deployer intelligence, and token risk scoring.

Auth is either of:
- ``MADEONSOL_API_KEY`` — an ``msk_`` API key (free tier available at
  https://madeonsol.com/pricing; free-tier live feeds are delayed 5 minutes,
  paid tiers are real-time).
- ``SVM_PRIVATE_KEY`` — a Solana private key for keyless x402 pay-per-call
  micropayments in USDC ($0.005-$0.02 per request, always real-time).
"""

import json
import os
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

try:
    from madeonsol_x402 import MadeOnSolClient

    MADEONSOL_AVAILABLE = True
except ImportError:
    MADEONSOL_AVAILABLE = False
    MadeOnSolClient = Any


def _make_client() -> "MadeOnSolClient":
    """Build a client from env vars. Priority: MADEONSOL_API_KEY > SVM_PRIVATE_KEY."""
    if not MADEONSOL_AVAILABLE:
        raise ImportError(
            "madeonsol-x402 is not installed. Install it with: "
            "uv add madeonsol-x402  (or pip install madeonsol-x402)"
        )
    api_key = os.environ.get("MADEONSOL_API_KEY", "")
    private_key = os.environ.get("SVM_PRIVATE_KEY", "")
    if api_key:
        return MadeOnSolClient(api_key=api_key)
    if private_key:
        return MadeOnSolClient(private_key=private_key)
    raise ValueError(
        "Set MADEONSOL_API_KEY (free key at https://madeonsol.com/pricing) or "
        "SVM_PRIVATE_KEY (x402 pay-per-call) to use the MadeOnSol tools."
    )


_MADEONSOL_ENV_VARS: List[EnvVar] = [
    EnvVar(
        name="MADEONSOL_API_KEY",
        description="MadeOnSol API key (msk_...). Free tier at https://madeonsol.com/pricing.",
        required=False,
    ),
    EnvVar(
        name="SVM_PRIVATE_KEY",
        description="Solana private key for keyless x402 pay-per-call (USDC micropayments). Alternative to MADEONSOL_API_KEY.",
        required=False,
    ),
]


class MadeOnSolKolFeedToolSchema(BaseModel):
    """Input for MadeOnSolKolFeedTool."""

    limit: int = Field(default=10, description="Number of trades to return (1-100).")
    action: Optional[str] = Field(
        default=None, description="Optional filter: 'buy' or 'sell'."
    )


class MadeOnSolKolFeedTool(BaseTool):
    """Live Solana KOL (key opinion leader) trade feed from MadeOnSol."""

    name: str = "MadeOnSol KOL Feed"
    description: str = (
        "Get the latest Solana KOL trades from 2,000+ tracked influencer wallets. "
        "Returns token, side, SOL amount, price, and the KOL's identity per trade."
    )
    args_schema: Type[BaseModel] = MadeOnSolKolFeedToolSchema
    env_vars: List[EnvVar] = _MADEONSOL_ENV_VARS
    package_dependencies: List[str] = Field(default_factory=lambda: ["madeonsol-x402"])

    def _run(self, limit: int = 10, action: Optional[str] = None) -> str:
        data = _make_client().kol_feed(limit=limit, action=action)
        return json.dumps(data, indent=2)


class MadeOnSolKolLeaderboardToolSchema(BaseModel):
    """Input for MadeOnSolKolLeaderboardTool."""

    period: str = Field(
        default="7d", description="Time period: today, 7d, 30d, 90d, or 180d."
    )
    limit: int = Field(default=10, description="Number of KOLs to return (1-50).")


class MadeOnSolKolLeaderboardTool(BaseTool):
    """Solana KOL performance rankings from MadeOnSol."""

    name: str = "MadeOnSol KOL Leaderboard"
    description: str = (
        "Get Solana KOL performance rankings by realized PnL and win rate over a "
        "chosen period. Useful for finding which influencer wallets are actually profitable."
    )
    args_schema: Type[BaseModel] = MadeOnSolKolLeaderboardToolSchema
    env_vars: List[EnvVar] = _MADEONSOL_ENV_VARS
    package_dependencies: List[str] = Field(default_factory=lambda: ["madeonsol-x402"])

    def _run(self, period: str = "7d", limit: int = 10) -> str:
        data = _make_client().kol_leaderboard(period=period, limit=limit)
        return json.dumps(data, indent=2)


class MadeOnSolKolCoordinationToolSchema(BaseModel):
    """Input for MadeOnSolKolCoordinationTool."""

    period: str = Field(default="24h", description="Time period: 1h, 6h, 24h, or 7d.")
    min_kols: int = Field(
        default=3, description="Minimum number of KOLs converging on a token (2-50)."
    )


class MadeOnSolKolCoordinationTool(BaseTool):
    """KOL convergence signals — tokens multiple KOLs are accumulating."""

    name: str = "MadeOnSol KOL Coordination"
    description: str = (
        "Get KOL convergence signals: tokens that multiple tracked KOL wallets are "
        "buying in the same window, with per-token KOL lists and buy totals."
    )
    args_schema: Type[BaseModel] = MadeOnSolKolCoordinationToolSchema
    env_vars: List[EnvVar] = _MADEONSOL_ENV_VARS
    package_dependencies: List[str] = Field(default_factory=lambda: ["madeonsol-x402"])

    def _run(self, period: str = "24h", min_kols: int = 3) -> str:
        data = _make_client().kol_coordination(period=period, min_kols=min_kols)
        return json.dumps(data, indent=2)


class MadeOnSolDeployerAlertsToolSchema(BaseModel):
    """Input for MadeOnSolDeployerAlertsTool."""

    limit: int = Field(default=10, description="Number of alerts to return (1-100).")
    tier: Optional[str] = Field(
        default=None,
        description=(
            "Optional deployer-tier filter: elite, good, moderate, rising, or cold "
            "(paid MadeOnSol tiers only)."
        ),
    )


class MadeOnSolDeployerAlertsTool(BaseTool):
    """Pump.fun deployer launch alerts with deployer track records."""

    name: str = "MadeOnSol Deployer Alerts"
    description: str = (
        "Get Pump.fun token-launch alerts enriched with the deployer's historical "
        "track record (prior launches, bond rate, rug history) and early KOL buys."
    )
    args_schema: Type[BaseModel] = MadeOnSolDeployerAlertsToolSchema
    env_vars: List[EnvVar] = _MADEONSOL_ENV_VARS
    package_dependencies: List[str] = Field(default_factory=lambda: ["madeonsol-x402"])

    def _run(self, limit: int = 10, tier: Optional[str] = None) -> str:
        data = _make_client().deployer_alerts(limit=limit, tier=tier)
        return json.dumps(data, indent=2)


class MadeOnSolTokenRiskToolSchema(BaseModel):
    """Input for MadeOnSolTokenRiskTool."""

    mint: str = Field(description="Solana token mint address (base58).")


class MadeOnSolTokenRiskTool(BaseTool):
    """Risk assessment for a Solana token."""

    name: str = "MadeOnSol Token Risk"
    description: str = (
        "Get a risk assessment for a Solana token by mint address: deployer "
        "reputation, sniper/bundler concentration, holder distribution, and "
        "dump-cluster signals."
    )
    args_schema: Type[BaseModel] = MadeOnSolTokenRiskToolSchema
    env_vars: List[EnvVar] = _MADEONSOL_ENV_VARS
    package_dependencies: List[str] = Field(default_factory=lambda: ["madeonsol-x402"])

    def _run(self, mint: str) -> str:
        data = _make_client().token_risk(mint)
        return json.dumps(data, indent=2)
