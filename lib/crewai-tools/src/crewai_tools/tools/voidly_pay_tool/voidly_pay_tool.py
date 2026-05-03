"""VoidlyPay tool wrapper for CrewAI.

Lets a CrewAI agent call any HTTP endpoint that speaks the
[x402 protocol](https://x402.org) and pay for it automatically when the
server returns ``HTTP 402 Payment Required``. Wraps the official
[`voidly-pay-crewai`](https://pypi.org/project/voidly-pay-crewai/) PyPI
package — when that package is installed the tool delegates to its
`VoidlyPayFetchTool` (full toolkit available via `VoidlyPayToolkit`).
Falls back to a self-contained x402 round-trip using `voidly-pay`
directly so the import never breaks at install time.

Mirrors the ``ComposioTool`` / ``LinkupSearchTool`` shape: a single
``BaseTool`` subclass with a Pydantic args schema that CrewAI can
serialise.

Install::

    pip install voidly-pay voidly-pay-crewai

Use::

    from crewai_tools import VoidlyPayTool
    from crewai import Agent

    agent = Agent(
        role="Researcher",
        goal="Pay for premium APIs without managing API keys.",
        tools=[VoidlyPayTool()],
    )
"""

from __future__ import annotations

import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


class VoidlyPayToolSchema(BaseModel):
    """Args for VoidlyPayTool."""

    url: str = Field(
        ...,
        description=(
            "HTTP URL to fetch. If the server returns 402 with a Voidly Pay "
            "x402 quote, the tool transfers the asking amount and retries."
        ),
    )
    method: str = Field(
        default="GET",
        description="HTTP method. Defaults to GET.",
    )
    body: Any | None = Field(
        default=None,
        description="JSON-serialisable body for POST/PUT.",
    )
    max_amount_credits: float = Field(
        default=0.05,
        description=(
            "Cap (in Voidly credits, where 1 credit = $0.001 USDC) on a "
            "single auto-pay. Default 0.05 = $0.05."
        ),
    )


class VoidlyPayTool(BaseTool):
    """Pay-on-402 fetch via the Voidly Pay rail.

    Settles via a Sourcify-verified USDC vault on Base mainnet
    (`0xb592512932a7b354969bb48039c2dc7ad6ad1c12`). Identity is loaded
    from the Voidly Pay SDK's local keypair file (env var
    ``VOIDLY_PAY_DID`` + ``VOIDLY_PAY_SECRET`` for explicit overrides).
    """

    name: str = "voidly_pay_fetch"
    description: str = (
        "Fetch any URL. If the server returns HTTP 402 with a Voidly Pay "
        "x402 quote, the tool auto-transfers credits and retries. Use this "
        "for paid endpoints behind an x402 paywall."
    )
    args_schema: type[BaseModel] = VoidlyPayToolSchema

    env_vars: list[EnvVar] = [
        EnvVar(
            name="VOIDLY_PAY_DID",
            description=(
                "DID of a funded Voidly Pay wallet (e.g. 'did:voidly:...'). "
                "If unset, the SDK loads ~/.voidly-pay/keypair.json. Mint a "
                "DID + claim the faucet at https://voidly.ai/pay/claim."
            ),
            required=False,
        ),
        EnvVar(
            name="VOIDLY_PAY_SECRET",
            description=(
                "base64-encoded Ed25519 secret key for VOIDLY_PAY_DID. "
                "Treated as sensitive — never logged."
            ),
            required=False,
        ),
        EnvVar(
            name="VOIDLY_PAY_API_BASE",
            description=("Override the rail URL. Defaults to https://api.voidly.ai."),
            required=False,
        ),
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _client(self) -> Any:
        """Return a VoidlyPay SDK client, lazily."""
        try:
            from voidly_pay import VoidlyPay  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "VoidlyPayTool requires the `voidly-pay` package. "
                "Install with: pip install voidly-pay"
            ) from e

        kwargs: dict = {}
        api_base = os.environ.get("VOIDLY_PAY_API_BASE")
        if api_base:
            kwargs["api_base"] = api_base
        did = os.environ.get("VOIDLY_PAY_DID")
        secret = os.environ.get("VOIDLY_PAY_SECRET")
        if did and secret:
            kwargs["did"] = did
            kwargs["secret_base64"] = secret
        return VoidlyPay(**kwargs)

    def _run(
        self,
        url: str,
        method: str = "GET",
        body: Any = None,
        max_amount_credits: float = 0.05,
    ) -> str:
        # Prefer the maintained crewAI integration when present — it has
        # a more complete 402-handler + receipt verification path.
        try:
            from voidly_pay_crewai import (
                VoidlyPayFetchTool,  # type: ignore[import-not-found]
            )

            return VoidlyPayFetchTool()._run(
                url=url,
                method=method,
                body=body,
                max_amount_credits=max_amount_credits,
            )
        except ImportError:
            pass  # fall through to inline implementation

        return _inline_x402_fetch(
            self._client(),
            url=url,
            method=method,
            body=body,
            max_amount_credits=max_amount_credits,
        )


def _inline_x402_fetch(
    pay: Any,
    url: str,
    method: str,
    body: Any,
    max_amount_credits: float,
) -> str:
    """Self-contained x402 round-trip — used when voidly-pay-crewai is absent.

    1. Make the request. If non-402, return the body.
    2. Parse the 402 envelope. Find the ``voidly-credit`` accept option.
    3. If asking price > cap, return a guarded refusal.
    4. Sign + submit the transfer to the recipient DID.
    5. Retry with ``?quote_id=<id>`` and return the resolved response.
    """
    import requests

    cap_micro = round(max_amount_credits * 1_000_000)

    def _do(extra_url: str = url) -> requests.Response:
        if body is not None:
            return requests.request(method, extra_url, json=body, timeout=30)
        return requests.request(method, extra_url, timeout=30)

    initial = _do()
    if initial.status_code != 402:
        return _stringify(initial)

    try:
        envelope = initial.json()
    except Exception:
        return _stringify(initial)

    accept = next(
        (a for a in envelope.get("accepts", []) if a.get("scheme") == "voidly-credit"),
        None,
    )
    if accept is None:
        return json.dumps(
            {"error": "no voidly-credit option in 402 envelope", "envelope": envelope},
            default=str,
        )

    if accept["amount_micro"] > cap_micro:
        return json.dumps(
            {
                "blocked_by_cap": True,
                "asked_micro": accept["amount_micro"],
                "cap_micro": cap_micro,
                "hint": ("Raise max_amount_credits if you want to pay this endpoint."),
            }
        )

    receipt = pay.pay(
        to=accept["recipient_did"],
        amount_micro=accept["amount_micro"],
        memo=f"x402:{accept['quote_id']}",
    )

    sep = "&" if "?" in url else "?"
    final = _do(f"{url}{sep}quote_id={accept['quote_id']}")
    return json.dumps(
        {
            "settled": True,
            "transfer_id": receipt.get("transfer_id"),
            "amount_micro": accept["amount_micro"],
            "recipient": accept["recipient_did"],
            "response_status": final.status_code,
            "response": _maybe_json(final),
        },
        default=str,
    )


def _stringify(r: Any) -> str:
    return json.dumps({"status": r.status_code, "body": _maybe_json(r)}, default=str)


def _maybe_json(r: Any) -> Any:
    ct = r.headers.get("content-type", "") if hasattr(r, "headers") else ""
    if ct.startswith("application/json"):
        try:
            return r.json()
        except (ValueError, TypeError):
            return r.text[:4000]
    return r.text[:4000]
