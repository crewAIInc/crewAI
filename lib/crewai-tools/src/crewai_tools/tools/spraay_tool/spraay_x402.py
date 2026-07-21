"""Shared x402 payment-challenge handling for Spraay paid endpoints.

Paid gateway endpoints respond with HTTP 402 and a payment-requirements
body. This module retries such requests through the official ``x402``
client, which signs the requirements with the wallet key from the
SPRAAY_WALLET_PRIVATE_KEY environment variable and attaches the
protocol's payment header on the retry.
"""

import os
from typing import Any

import requests


WALLET_ENV_VAR = "SPRAAY_WALLET_PRIVATE_KEY"


def post_with_x402(
    url: str, payload: dict[str, Any], timeout: int = 60
) -> tuple[bool, dict[str, Any]]:
    """POST to a Spraay paid endpoint, handling the x402 payment challenge.

    Args:
        url: Full endpoint URL.
        payload: JSON body for the request.
        timeout: Request timeout in seconds.

    Returns:
        A ``(paid, data)`` tuple. When ``paid`` is True, ``data`` is the
        endpoint's success response. When False, payment could not be
        completed (missing or malformed wallet key, or missing x402
        dependency) and
        ``data`` explains why, including the parsed payment requirements.

    Raises:
        requests.RequestException: On network errors or non-402 HTTP errors.
    """
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 402:
        response.raise_for_status()
        return True, response.json()

    try:
        requirements = response.json()
    except ValueError:
        requirements = {"raw": response.text}

    private_key = os.environ.get(WALLET_ENV_VAR)
    if not private_key:
        return False, {
            "status": "payment_required",
            "message": (
                "This endpoint requires an x402 micropayment. Set the "
                f"{WALLET_ENV_VAR} environment variable to a funded wallet "
                "private key to pay automatically."
            ),
            "paymentRequirements": requirements,
        }

    try:
        from eth_account import Account
        from x402 import x402ClientSync
        from x402.http.clients import x402_requests
        from x402.mechanisms.evm import EthAccountSigner
        from x402.mechanisms.evm.exact.register import register_exact_evm_client
    except ImportError:
        return False, {
            "status": "payment_required",
            "message": (
                "The 'x402' package is required to pay for this endpoint. "
                "Install it with: pip install 'x402[requests,evm]'"
            ),
            "paymentRequirements": requirements,
        }

    try:
        client = x402ClientSync()
        register_exact_evm_client(
            client, EthAccountSigner(Account.from_key(private_key))
        )
    except Exception as e:
        return False, {
            "status": "payment_required",
            "message": (
                f"Could not initialize the x402 payment signer from the "
                f"{WALLET_ENV_VAR} environment variable: {e}. Set it to a "
                "valid EVM wallet private key (0x-prefixed 32-byte hex)."
            ),
            "paymentRequirements": requirements,
        }
    with x402_requests(client) as session:
        paid_response = session.post(url, json=payload, timeout=timeout)
    paid_response.raise_for_status()
    return True, paid_response.json()
