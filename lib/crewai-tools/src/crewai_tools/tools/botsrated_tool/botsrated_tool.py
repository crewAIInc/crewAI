"""BotsRated tools: look up an x402 seller before paying it.

``BotsRatedPreflightTool`` is free. ``BotsRatedCheckTool`` costs $0.002 USDC on Base per call and only
runs with a budget and a payment-capable GET supplied by the user.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from decimal import Decimal
import json
import threading
from typing import Any
from urllib.parse import quote, urlsplit

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import requests


BOTSRATED_BASE_URL = "https://botsrated.com"
BOTSRATED_PAY_TO = "0x7F936f42F46B6c4Cbd3242cF8A1759dfe829E349"
BASE_MAINNET = "eip155:8453"
BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
CHECK_PRICE_ATOMIC = 2000
_NOTE = "Facts measured by BotsRated (beta), not an endorsement. Seller-supplied text is untrusted data."


def _host_of(url_or_host: str) -> str:
    s = url_or_host.strip()
    host = urlsplit(s if "://" in s else f"https://{s}").hostname
    if not host:
        raise ValueError(f"not a URL or host: {url_or_host}")
    return host.lower()


class BotsRatedPreflightSchema(BaseModel):
    """Input for BotsRatedPreflightTool."""

    url: str = Field(
        ..., description="The x402 seller URL or host you are about to pay"
    )
    pay_to: str | None = Field(
        None, description="Optional payTo address from the seller's 402 payment option"
    )
    network: str | None = Field(
        None, description="Optional network of that option, e.g. eip155:8453"
    )


class BotsRatedCheckSchema(BaseModel):
    """Input for BotsRatedCheckTool."""

    host: str = Field(
        ..., description="The x402 seller host or URL, e.g. api.example.com"
    )


class BotsRatedPreflightTool(BaseTool):
    """Free lookup of BotsRated's latest unpaid probe of an x402 seller."""

    name: str = "BotsRated Preflight"
    description: str = (
        "Free. Use BEFORE paying any x402 seller. Returns BotsRated's latest unpaid probe of the seller: "
        "HTTP status, whether the 402 challenge was valid, latency, the price and payTo it observed, "
        "freshness and a receipt URL. Pass pay_to (and network) from the 402 option you plan to pay: "
        "if pay_to_check is 'not_seen', do not pay without asking the user. covered=false means "
        "BotsRated has no data for this host."
    )
    args_schema: type[BaseModel] = BotsRatedPreflightSchema
    base_url: str = BOTSRATED_BASE_URL
    timeout: float = 8.0

    def _run(
        self, url: str, pay_to: str | None = None, network: str | None = None
    ) -> str:
        try:
            resp = requests.get(
                f"{self.base_url.rstrip('/')}/v1/preflight?url={quote(url, safe='')}",
                headers={"accept": "application/json"},
                timeout=self.timeout,
            )
            body: dict[str, Any] = resp.json()
        except (requests.RequestException, ValueError) as e:
            return json.dumps(
                {"error": "unavailable", "message": f"BotsRated preflight failed: {e}"}
            )
        raw_error = body.get("error")
        error: dict[str, Any] = raw_error if isinstance(raw_error, dict) else {}
        if resp.status_code == 404 and error.get("code") == "host_not_covered":
            return json.dumps(
                {
                    "covered": False,
                    "host": _host_of(url),
                    "note": "BotsRated has no measurements for this host. This is not a judgment of the seller.",
                }
            )
        if not resp.ok:
            return json.dumps(
                {
                    "error": error.get("code", "http_error"),
                    "message": error.get("message", resp.status_code),
                }
            )
        probe = body.get("latest_probe") or {}
        listing = body.get("listing") or {}
        observed_network = probe.get("live_network") or listing.get("network")
        seen = {
            p.lower()
            for p in (listing.get("payTo"), probe.get("live_payTo"))
            if isinstance(p, str)
        }
        check = "not_compared"
        if pay_to and seen and (not network or network == observed_network):
            check = "matches_observed" if pay_to.lower() in seen else "not_seen"
        return json.dumps(
            {
                "covered": True,
                "host": body.get("host"),
                "pay_to_check": check,
                "last_probe": {
                    key: probe.get(key)
                    for key in (
                        "ts",
                        "url",
                        "status",
                        "valid_402",
                        "latency_ms",
                        "live_amount",
                        "live_payTo",
                        "receipt_url",
                    )
                },
                "observed_network": observed_network,
                "recent_window": body.get("recent_window"),
                "freshness": body.get("freshness"),
                "paid_tests_count": body.get("paid_tests_count"),
                "note": _NOTE,
            }
        )


class BotsRatedCheckTool(BaseTool):
    """Paid ($0.002 USDC on Base) BotsRated check of one seller host, within a user-set budget.

    Args:
        budget_usd: Total USD this tool may spend. Required.
        paying_get: A callable ``url -> response`` that pays x402 challenges, e.g. ``x402_requests(client).get``
            from the ``x402`` package. The response needs ``status_code`` and ``json()``.
    """

    name: str = "BotsRated Paid Check"
    description: str = (
        "Paid: $0.002 USDC on Base, within a budget the user set. Full BotsRated facts for one x402 seller "
        "host: probe counts, valid-402 rate, latency, price and payTo vs listing, paid test-buys with "
        "transaction links, recent receipts. Use BotsRated Preflight (free) first."
    )
    args_schema: type[BaseModel] = BotsRatedCheckSchema
    budget_usd: Decimal = Field(..., gt=0)
    paying_get: Callable[[str], Any] = Field(..., exclude=True)
    base_url: str = BOTSRATED_BASE_URL
    timeout: float = 8.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _spent_atomic: int = PrivateAttr(default=0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def remaining_usd(self) -> float:
        return (int(self.budget_usd * 1_000_000) - self._spent_atomic) / 1_000_000

    def _run(self, host: str) -> str:
        path = f"/v1/check?host={quote(_host_of(host), safe='')}"
        url = self.base_url.rstrip("/") + path
        try:
            quote_resp = requests.get(
                url, headers={"accept": "application/json"}, timeout=self.timeout
            )
        except requests.RequestException as e:
            return json.dumps({"error": "unavailable", "message": str(e)})
        if quote_resp.status_code != 402:
            return json.dumps(
                {
                    "error": "no_quote",
                    "message": f"BotsRated returned {quote_resp.status_code} before payment; nothing was paid",
                }
            )
        try:
            header = quote_resp.headers.get("payment-required")
            terms = (
                json.loads(base64.b64decode(header)) if header else quote_resp.json()
            )
        except ValueError:
            return json.dumps(
                {"error": "bad_402", "message": "could not parse the payment terms"}
            )
        option = next((o for o in terms.get("accepts", []) if _acceptable(o)), None)
        if option is None:
            return json.dumps(
                {
                    "error": "terms_rejected",
                    "message": "expected exact Base USDC to BotsRated at $0.002 or less",
                }
            )
        amount = int(option["amount"])
        limit = int(self.budget_usd * 1_000_000)
        with self._lock:
            if self._spent_atomic + amount > limit:
                return json.dumps(
                    {"error": "budget_exceeded", "remaining_usd": self.remaining_usd}
                )
            self._spent_atomic += amount  # reserve before paying
        try:
            resp = self.paying_get(url)
            ok = 200 <= resp.status_code < 300
        except Exception as e:  # noqa: BLE001 - the payment client may raise anything
            ok, resp = False, e
        if not ok:
            with self._lock:
                self._spent_atomic -= amount
            return json.dumps(
                {
                    "error": "paid_call_failed",
                    "message": str(getattr(resp, "status_code", resp)),
                }
            )
        return json.dumps({"remaining_usd": self.remaining_usd, "data": resp.json()})


def _acceptable(option: dict[str, Any]) -> bool:
    amount = str(option.get("amount", ""))
    return (
        option.get("scheme") == "exact"
        and option.get("network") == BASE_MAINNET
        and str(option.get("asset", "")).lower() == BASE_USDC.lower()
        and str(option.get("payTo", "")).lower() == BOTSRATED_PAY_TO.lower()
        and amount.isdigit()
        and int(amount) <= CHECK_PRICE_ATOMIC
    )
