"""A minimal x402 v2 payment client for StallRow tools.

Implements the 402 -> sign -> retry flow of the x402 "exact" scheme on
Base (EIP-3009 `TransferWithAuthorization`), the same flow the fleet's
own production clients run against StallRow:

1. POST the endpoint unpaid -> HTTP 402 with a base64 JSON challenge in
   the `payment-required` response header (also mirrored in the body).
2. Pick the `exact` / `eip155:8453` entry from `accepts`.
3. Enforce the asset (USDC on Base), the per-call price cap, and the
   authorization validity-window ceiling *before* anything is signed.
4. Build the EIP-3009 authorization and sign it as EIP-712 typed data
   with the domain the challenge advertises in `accepts.extra`
   (for USDC on Base: name "USD Coin", version "2").
5. Retry the identical POST with the payment payload in the
   `PAYMENT-SIGNATURE` header (base64 JSON). The service's facilitator
   verifies and settles USDC on Base server-side.

Safety contract: with no payer key configured this client NEVER pays —
the caller gets a `X402NotConfiguredError` naming the price it *would*
have paid. The key is read from the environment only and is never
logged. The `eth_account` signer dependency is imported lazily, at the
moment a payment is actually attempted, so importing this module (and
the tool built on it) works without it installed.
"""

from __future__ import annotations

import base64
from decimal import Decimal
import json
import os
import secrets
import time
from typing import Any

import requests


BASE_NETWORK = "eip155:8453"
BASE_USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_DECIMALS = 6
DEFAULT_MAX_PRICE_USDC = "0.005"
DEFAULT_TIMEOUT_SECONDS = 300
# Client-side ceiling on the EIP-3009 validity window: the endpoint
# chooses maxTimeoutSeconds, but this client never signs an
# authorization that stays settleable for longer than this, however
# long the challenge asks for.
MAX_TIMEOUT_SECONDS = 300
PRIVATE_KEY_ENV_VAR = "STALLROW_WALLET_PRIVATE_KEY"
MAX_PRICE_ENV_VAR = "STALLROW_MAX_PRICE_USDC"

TRANSFER_WITH_AUTHORIZATION_TYPES = {
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ]
}


class X402Error(Exception):
    """Base class for x402 payment failures."""


class X402NotConfiguredError(X402Error):
    """A payment was required but no payer key is configured."""


class X402PriceCapExceededError(X402Error):
    """The challenged price exceeds the configured per-call cap."""


class X402UnsupportedAssetError(X402Error):
    """The challenged asset is not USDC on Base."""


class X402PaymentError(X402Error):
    """The payment flow failed (bad challenge, rejected payment, ...)."""


def parse_payment_required(response: requests.Response) -> dict[str, Any]:
    """Parse the x402 v2 challenge from a 402 response.

    The challenge travels base64-encoded in the `payment-required`
    header; some stacks also put it in the JSON body. Raises
    `X402PaymentError` when neither carries a usable challenge.
    """
    header = response.headers.get("payment-required")
    if header:
        try:
            decoded = json.loads(base64.b64decode(header).decode("utf-8"))
            if isinstance(decoded, dict) and decoded.get("accepts"):
                return decoded
        except (ValueError, UnicodeDecodeError):
            pass
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("accepts"):
        return body
    raise X402PaymentError("402 response carried no parseable x402 payment challenge")


def select_requirement(
    challenge: dict[str, Any], network: str = BASE_NETWORK, scheme: str = "exact"
) -> dict[str, Any]:
    """Pick the accepts entry this client can settle (exact EVM on Base)."""
    accepts: list[dict[str, Any]] = challenge.get("accepts") or []
    for entry in accepts:
        if entry.get("scheme") == scheme and entry.get("network") == network:
            return entry
    offered = [(a.get("scheme"), a.get("network")) for a in accepts]
    raise X402PaymentError(
        f"challenge offers no {scheme} requirement on {network}: {offered}"
    )


def build_authorization(
    from_address: str,
    requirement: dict[str, Any],
    *,
    now: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """EIP-3009 authorization, string fields exactly as the reference
    x402 EVM clients build them.

    The challenge's `maxTimeoutSeconds` is refused, before anything is
    signed, when it is unusable or exceeds `MAX_TIMEOUT_SECONDS`: the
    endpoint alone must not be able to make a signed authorization
    settleable for months or years after the call that produced it.
    """
    now = int(time.time()) if now is None else now
    try:
        timeout = int(requirement.get("maxTimeoutSeconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError) as e:
        raise X402PaymentError(
            "payment requirement carries no usable maxTimeoutSeconds: "
            f"{requirement.get('maxTimeoutSeconds')!r}"
        ) from e
    if timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
        raise X402PaymentError(
            f"endpoint requires a payment validity window of {timeout} "
            f"seconds, outside the 1-{MAX_TIMEOUT_SECONDS} second window "
            "this client will sign"
        )
    return {
        "from": from_address,
        "to": requirement["payTo"],
        "value": str(requirement["amount"]),
        "validAfter": "0",
        "validBefore": str(now + timeout),
        "nonce": nonce or ("0x" + secrets.token_hex(32)),
    }


def sign_authorization(
    account: Any, requirement: dict[str, Any], authorization: dict[str, str]
) -> str:
    """EIP-712 sign the TransferWithAuthorization against the token
    domain the challenge advertises in accepts.extra (name/version).

    `account` is an eth_account LocalAccount.
    """
    extra = requirement.get("extra") or {}
    if not extra.get("name") or not extra.get("version"):
        raise X402PaymentError(
            "payment requirement is missing EIP-712 domain parameters "
            f"(extra.name/extra.version) for asset {requirement.get('asset')}"
        )
    network = requirement.get("network", BASE_NETWORK)
    if not network.startswith("eip155:"):
        raise X402PaymentError(f"unsupported network for EVM signing: {network}")
    domain = {
        "name": extra["name"],
        "version": extra["version"],
        "chainId": int(network.split(":", 1)[1]),
        "verifyingContract": requirement["asset"],
    }
    message = {
        "from": authorization["from"],
        "to": authorization["to"],
        "value": int(authorization["value"]),
        "validAfter": int(authorization["validAfter"]),
        "validBefore": int(authorization["validBefore"]),
        "nonce": bytes.fromhex(authorization["nonce"].removeprefix("0x")),
    }
    signed = account.sign_typed_data(
        domain_data=domain,
        message_types=TRANSFER_WITH_AUTHORIZATION_TYPES,
        message_data=message,
    )
    hex_sig: str = signed.signature.hex()
    return hex_sig if hex_sig.startswith("0x") else "0x" + hex_sig


def build_payment_payload(
    challenge: dict[str, Any],
    requirement: dict[str, Any],
    authorization: dict[str, str],
    signature: str,
) -> dict[str, Any]:
    """The x402 v2 PaymentPayload the paid retry carries.

    The challenge's `extensions` (notably the Bazaar discovery
    declaration) are echoed into the payload: facilitator catalogues
    index a resource from the payment payload's bazaar extension, not
    from the seller's 402.
    """
    payload: dict[str, Any] = {
        "x402Version": challenge.get("x402Version", 2),
        "resource": challenge.get("resource"),
        "accepted": requirement,
        "payload": {"authorization": authorization, "signature": signature},
    }
    extensions = challenge.get("extensions")
    if extensions:
        payload["extensions"] = extensions
    return payload


def encode_payment_header(payload: dict[str, Any]) -> str:
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


class X402Client:
    """Pays x402 v2 challenges for JSON POST endpoints, under a hard cap.

    `private_key` is the payer's EVM private key (0x-prefixed). It is
    only ever used to sign EIP-3009 authorizations locally; it is never
    sent anywhere or logged. When it is None, any endpoint that answers
    402 fails closed with `X402NotConfiguredError`.
    """

    def __init__(
        self,
        private_key: str | None = None,
        max_price_usdc: str | Decimal = DEFAULT_MAX_PRICE_USDC,
    ) -> None:
        self._private_key = private_key.strip() if private_key else None
        self.max_price_usdc = Decimal(str(max_price_usdc))

    @classmethod
    def from_env(cls, max_price_usdc: str | Decimal | None = None) -> X402Client:
        """Build a client from the environment.

        Key: `STALLROW_WALLET_PRIVATE_KEY`. Cap: the explicit argument
        wins, then `STALLROW_MAX_PRICE_USDC`, then the $0.005 default.
        """
        key = (os.environ.get(PRIVATE_KEY_ENV_VAR) or "").strip() or None
        cap = max_price_usdc
        if cap is None:
            cap = (os.environ.get(MAX_PRICE_ENV_VAR) or "").strip() or None
        if cap is None:
            cap = DEFAULT_MAX_PRICE_USDC
        return cls(private_key=key, max_price_usdc=cap)

    @property
    def max_price_units(self) -> int:
        """The cap in USDC base units (6 decimals)."""
        return int(self.max_price_usdc * (10**USDC_DECIMALS))

    def post_json(
        self, url: str, payload: dict[str, Any], *, timeout: int = 30
    ) -> dict[str, Any]:
        """POST `payload` as JSON, paying the x402 challenge if asked.

        Returns the endpoint's JSON body on success. Raises X402Error
        subclasses for every payment-side failure; requests exceptions
        propagate for transport failures.
        """
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return self._json_body(response)
        if response.status_code != 402:
            raise X402PaymentError(
                f"request failed: HTTP {response.status_code}: {response.text[:500]}"
            )

        challenge = parse_payment_required(response)
        requirement = select_requirement(challenge)
        try:
            amount_units = int(requirement["amount"])
        except (KeyError, TypeError, ValueError) as e:
            raise X402PaymentError(
                f"payment requirement carries no usable amount: {requirement!r}"
            ) from e

        # The asset and the cap are enforced before the key is even
        # looked at: a non-USDC asset or an over-cap price is refused
        # without signing anything.
        asset = str(requirement.get("asset") or "")
        if asset.lower() != BASE_USDC_ADDRESS.lower():
            raise X402UnsupportedAssetError(
                f"endpoint requires payment in asset {asset or '<missing>'}, "
                f"but this client only settles USDC on Base ({BASE_USDC_ADDRESS})"
            )
        if amount_units > self.max_price_units:
            raise X402PriceCapExceededError(
                f"endpoint requires ${Decimal(amount_units) / (10**USDC_DECIMALS)} "
                f"USDC, above the per-call cap of ${self.max_price_usdc} USDC "
                f"(raise {MAX_PRICE_ENV_VAR} to allow it)"
            )
        if not self._private_key:
            raise X402NotConfiguredError(
                f"endpoint requires a payment of "
                f"${Decimal(amount_units) / (10**USDC_DECIMALS)} USDC but no "
                f"payer wallet is configured; set {PRIVATE_KEY_ENV_VAR} to a "
                "funded Base wallet's private key to pay x402 endpoints"
            )

        header = self._payment_header(challenge, requirement)
        paid = requests.post(
            url,
            json=payload,
            headers={"PAYMENT-SIGNATURE": header},
            timeout=timeout,
        )
        if paid.status_code == 402:
            raise X402PaymentError(
                "payment was not accepted: the endpoint answered 402 again "
                "after the paid retry"
            )
        if paid.status_code != 200:
            raise X402PaymentError(
                f"paid request failed: HTTP {paid.status_code}: {paid.text[:500]}"
            )
        return self._json_body(paid)

    def _payment_header(
        self, challenge: dict[str, Any], requirement: dict[str, Any]
    ) -> str:
        try:
            from eth_account import Account
        except ImportError:
            raise X402PaymentError(
                "the 'eth-account' package is required to sign x402 "
                "payments; install it with: pip install 'crewai-tools[stallrow]'"
            ) from None
        key = self._private_key or ""
        if not key.startswith("0x"):
            key = "0x" + key
        try:
            account = Account.from_key(key)
        except (ValueError, TypeError) as e:
            raise X402PaymentError(
                f"{PRIVATE_KEY_ENV_VAR} is not a usable EVM private key"
            ) from e
        authorization = build_authorization(account.address, requirement)
        signature = sign_authorization(account, requirement, authorization)
        payload = build_payment_payload(
            challenge, requirement, authorization, signature
        )
        return encode_payment_header(payload)

    @staticmethod
    def _json_body(response: requests.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as e:
            raise X402PaymentError(
                f"endpoint returned a non-JSON success body: {response.text[:200]}"
            ) from e
        if not isinstance(body, dict):
            raise X402PaymentError(
                f"endpoint returned an unexpected JSON body: {body!r}"
            )
        return body
