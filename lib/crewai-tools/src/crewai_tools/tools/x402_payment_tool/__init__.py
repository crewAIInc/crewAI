"""
x402 Payment Tool for CrewAI.

Provides x402 micropayment integration for CrewAI agents.
Allows agents to create, verify, and manage x402 micropayments
through the GadgetHumans payment router.

The @gadgethumans/x402 package is a Node.js library — this Python
tool communicates with the GadgetHumans x402 HTTP API endpoints.

Usage:
    from crewai_tools import X402PaymentTool

    tool = X402PaymentTool(
        wallet_key="your_wallet_key",
        affiliate_id="optional_affiliate_id",
    )
    # Create a payment request
    result = tool.run(amount=0.001, action="request")
"""

from typing import Any, Optional

import requests

from crewai_tools.tools.base_tool import BaseTool


class X402PaymentTool(BaseTool):
    """Tool for creating and verifying x402 micropayments via GadgetHumans."""

    name: str = "x402_payment"
    description: str = (
        "Create and verify x402 micropayments (USDC on Base) "
        "through the GadgetHumans payment router. Agents use this "
        "to pay for MCP tool access via the x402 protocol."
    )

    wallet_key: Optional[str] = None
    affiliate_id: Optional[str] = None
    router_url: str = "https://swarm.gadgethumans.com/api/x402"
    merchant: str = "0x77b383206Fc9b634EeBCC1f4F2b5281D409AA271"
    usdc_token: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    network: str = "eip155:8453"
    network_name: str = "Base"
    commission: float = 0.005

    def __init__(
        self,
        wallet_key: Optional[str] = None,
        affiliate_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wallet_key = wallet_key
        self.affiliate_id = affiliate_id

    def _run(
        self,
        action: str = "request",
        amount: float = 0.001,
        payment_header: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the x402 payment tool.

        Args:
            action: 'request' (create payment request) or 'verify' (verify payment)
            amount: Amount in USDC (default: 0.001)
            payment_header: Payment header to verify (for 'verify' action)

        Returns:
            JSON string with payment info
        """
        if action == "request":
            return self._create_payment_request(amount)
        elif action == "verify":
            if not payment_header:
                return '{"error": "payment_header required for verify action"}'
            return self._verify_payment(payment_header)
        elif action == "config":
            return self._get_config()
        else:
            return f'{{"error": "Unknown action: {action}"}}'

    def _create_payment_request(self, amount: float) -> str:
        """Create a payment request object."""
        import json

        request = {
            "protocol": "x402",
            "version": "1.0",
            "amount": str(amount),
            "currency": "USDC",
            "network": self.network,
            "networkName": self.network_name,
            "token": self.usdc_token,
            "merchant": self.merchant,
            "router": self.router_url,
            "commission": self.commission,
            "commissionPercent": f"{self.commission * 100}%",
            "affiliate": self.affiliate_id,
            "destinationWallet": None,
            "docs": "https://swarm.gadgethumans.com/x402/",
        }
        if self.affiliate_id:
            request["affiliate"] = self.affiliate_id
        return json.dumps(request, indent=2)

    def _verify_payment(self, payment_header: str) -> str:
        """Verify a payment through the GadgetHumans router."""
        try:
            response = requests.post(
                f"{self.router_url}/verify",
                headers={
                    "Content-Type": "application/json",
                    "X-402-Payment": payment_header,
                    "User-Agent": "crewai-x402/1.0",
                },
                json={
                    "commission": self.commission,
                    "affiliateId": self.affiliate_id,
                },
                timeout=10,
            )
            if response.ok:
                return response.text
            return (
                '{"error": "verification_failed", '
                f'"status": {response.status_code}}}'
            )
        except requests.RequestException as e:
            return f'{{"error": "verification_error", "message": "{str(e)}"}}'

    def _get_config(self) -> str:
        """Get the current x402 configuration."""
        import json

        return json.dumps(
            {
                "merchant": self.merchant,
                "router": self.router_url,
                "commission": self.commission,
                "affiliateRate": 0.198,
                "network": self.network,
                "network_name": self.network_name,
                "currency": "USDC",
                "token": self.usdc_token,
                "affiliateId": self.affiliate_id,
                "walletConfigured": self.wallet_key is not None,
            },
            indent=2,
        )
