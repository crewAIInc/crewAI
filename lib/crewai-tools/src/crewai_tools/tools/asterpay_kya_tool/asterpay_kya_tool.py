"""AsterPay KYA (Know Your Agent) trust scoring tool for CrewAI.

Check the trustworthiness of AI agents before transacting.
Free API — no key required. Docs: https://asterpay.io
"""

import json
from typing import Any, Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class AsterPayKYAInput(BaseModel):
    """Input for AsterPay KYA trust score lookup."""

    address: str = Field(
        description="Ethereum address (0x...) of the AI agent or wallet to score."
    )
    action: str = Field(
        default="trust_score",
        description=(
            "Action to perform: 'trust_score' (0-100 score), "
            "'verify' (ERC-8004 identity check), "
            "'tier' (Open/Verified/Trusted/Enterprise), or "
            "'settlement_estimate' (USDC to EUR estimate, use address field for amount)."
        ),
    )


class AsterPayKYATool(BaseTool):
    """Check AI agent trust scores via AsterPay KYA (Know Your Agent).

    Returns trust score (0-100), tier, ERC-8004 identity verification,
    and sanctions screening results. Free, no API key required.

    Supports x402, MPP (Stripe/Tempo), and AP2 (Google) payment protocols.
    """

    name: str = "AsterPay KYA Trust Score"
    description: str = (
        "Check the trust score (0-100) of an AI agent or wallet address "
        "before transacting. Returns trust tier (Open/Verified/Trusted/"
        "Enterprise), component scores (ERC-8004 identity, sanctions "
        "screening via Chainalysis, on-chain activity, behavioral signals), "
        "and risk assessment. Also supports agent identity verification "
        "and USDC to EUR settlement estimates. Free, no API key needed."
    )
    args_schema: Type[BaseModel] = AsterPayKYAInput
    base_url: str = "https://x402.asterpay.io"

    def _run(self, **kwargs: Any) -> str:
        address = kwargs.get("address", "")
        action = kwargs.get("action", "trust_score")

        endpoints = {
            "trust_score": f"/v1/kya/trust-score/{address}",
            "verify": f"/v1/kya/verify/{address}",
            "tier": f"/v1/kya/tier/{address}",
            "settlement_estimate": "/v1/settlement/estimate",
        }

        endpoint = endpoints.get(action, endpoints["trust_score"])
        url = f"{self.base_url}{endpoint}"

        try:
            if action == "settlement_estimate":
                try:
                    amount = float(address)
                except ValueError:
                    amount = 100.0
                response = requests.get(
                    url, params={"amount": amount}, timeout=30
                )
            else:
                response = requests.get(url, timeout=30)

            response.raise_for_status()
            return json.dumps(response.json(), indent=2, default=str)
        except Exception as e:
            return f"Error: {e}"
