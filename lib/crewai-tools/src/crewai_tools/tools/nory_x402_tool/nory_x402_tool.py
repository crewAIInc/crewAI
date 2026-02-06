"""Nory x402 Payment Tools for CrewAI.

Tools for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

import os
from typing import Any, List, Optional, Type

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


NORY_API_BASE = "https://noryx402.com"


class PaymentRequirementsSchema(BaseModel):
    """Input for getting payment requirements."""

    resource: str = Field(
        ..., description="The resource path requiring payment (e.g., /api/premium/data)"
    )
    amount: str = Field(
        ..., description="Amount in human-readable format (e.g., '0.10' for $0.10 USDC)"
    )
    network: Optional[str] = Field(
        None,
        description="Preferred network: solana-mainnet, base-mainnet, polygon-mainnet, etc.",
    )


class VerifyPaymentSchema(BaseModel):
    """Input for verifying a payment."""

    payload: str = Field(
        ..., description="Base64-encoded payment payload containing signed transaction"
    )


class SettlePaymentSchema(BaseModel):
    """Input for settling a payment."""

    payload: str = Field(..., description="Base64-encoded payment payload")


class TransactionLookupSchema(BaseModel):
    """Input for looking up a transaction."""

    transaction_id: str = Field(..., description="Transaction ID or signature")
    network: str = Field(
        ..., description="Network ID (e.g., solana-mainnet, base-mainnet)"
    )


class NoryPaymentRequirementsTool(BaseTool):
    """
    NoryPaymentRequirementsTool - Get x402 payment requirements for a resource.

    Use this when you encounter an HTTP 402 Payment Required response
    and need to know how much to pay and where to send payment.

    Returns the amount, supported networks, and wallet address needed
    to make a payment.
    """

    name: str = "Nory Get Payment Requirements"
    description: str = (
        "Get payment requirements for accessing a paid resource. "
        "Returns the amount, supported networks, and wallet address needed. "
        "Use when you encounter an HTTP 402 Payment Required response."
    )
    args_schema: Type[BaseModel] = PaymentRequirementsSchema
    env_vars: List[EnvVar] = [
        EnvVar(
            name="NORY_API_KEY",
            description="API key for Nory (optional for public endpoints)",
            required=False,
        ),
    ]

    def _run(self, **kwargs: Any) -> str:
        resource = kwargs.get("resource")
        amount = kwargs.get("amount")
        network = kwargs.get("network")

        if not resource or not amount:
            return "Error: resource and amount are required"

        params = {"resource": resource, "amount": amount}
        if network:
            params["network"] = network

        headers = {}
        api_key = os.environ.get("NORY_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.get(
                f"{NORY_API_BASE}/api/x402/requirements",
                params=params,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error getting payment requirements: {str(e)}"


class NoryVerifyPaymentTool(BaseTool):
    """
    NoryVerifyPaymentTool - Verify a payment before settlement.

    Use this to validate that a signed payment transaction is correct
    before submitting it to the blockchain.
    """

    name: str = "Nory Verify Payment"
    description: str = (
        "Verify a signed payment transaction before submitting to blockchain. "
        "Use to validate a payment is correct before settlement."
    )
    args_schema: Type[BaseModel] = VerifyPaymentSchema
    env_vars: List[EnvVar] = [
        EnvVar(
            name="NORY_API_KEY",
            description="API key for Nory (optional for public endpoints)",
            required=False,
        ),
    ]

    def _run(self, **kwargs: Any) -> str:
        payload = kwargs.get("payload")

        if not payload:
            return "Error: payload is required"

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("NORY_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.post(
                f"{NORY_API_BASE}/api/x402/verify",
                json={"payload": payload},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error verifying payment: {str(e)}"


class NorySettlePaymentTool(BaseTool):
    """
    NorySettlePaymentTool - Settle a payment on-chain.

    Use this to submit a verified payment transaction to the blockchain.
    Settlement typically completes in under 400ms.
    """

    name: str = "Nory Settle Payment"
    description: str = (
        "Submit a verified payment to the blockchain for settlement. "
        "Returns transaction ID upon success (~400ms settlement time)."
    )
    args_schema: Type[BaseModel] = SettlePaymentSchema
    env_vars: List[EnvVar] = [
        EnvVar(
            name="NORY_API_KEY",
            description="API key for Nory (optional for public endpoints)",
            required=False,
        ),
    ]

    def _run(self, **kwargs: Any) -> str:
        payload = kwargs.get("payload")

        if not payload:
            return "Error: payload is required"

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("NORY_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.post(
                f"{NORY_API_BASE}/api/x402/settle",
                json={"payload": payload},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error settling payment: {str(e)}"


class NoryTransactionLookupTool(BaseTool):
    """
    NoryTransactionLookupTool - Look up transaction status.

    Use this to check the status of a previously submitted payment.
    """

    name: str = "Nory Transaction Lookup"
    description: str = (
        "Look up the status and details of a transaction by ID or signature."
    )
    args_schema: Type[BaseModel] = TransactionLookupSchema
    env_vars: List[EnvVar] = [
        EnvVar(
            name="NORY_API_KEY",
            description="API key for Nory (optional for public endpoints)",
            required=False,
        ),
    ]

    def _run(self, **kwargs: Any) -> str:
        transaction_id = kwargs.get("transaction_id")
        network = kwargs.get("network")

        if not transaction_id or not network:
            return "Error: transaction_id and network are required"

        headers = {}
        api_key = os.environ.get("NORY_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.get(
                f"{NORY_API_BASE}/api/x402/transactions/{transaction_id}",
                params={"network": network},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error looking up transaction: {str(e)}"


class NoryHealthCheckTool(BaseTool):
    """
    NoryHealthCheckTool - Check Nory service health.

    Use this to verify the payment service is operational
    and see supported networks.
    """

    name: str = "Nory Health Check"
    description: str = (
        "Check health status of Nory x402 payment service and supported networks."
    )

    def _run(self, **kwargs: Any) -> str:
        try:
            response = requests.get(f"{NORY_API_BASE}/api/x402/health", timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error checking health: {str(e)}"
