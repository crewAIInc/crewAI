"""Typed client for the SINCOR platform API (standard library only).

Live today on https://getsincor.com: CNP auction tasks, escrowed task
creation, and pool settlement with a 5% platform fee. SEAL (EIP-712
settlement authorization) and SIGIL (state-commitment verification) are
integration-pilot surfaces; this client does not claim them as settled
mainnet primitives, and no settlement contract is deployed on Base
mainnet. The SINCOR token is platform access, not an investment.

The companion verified-payment package for x402 endpoints is published
separately as ``sincor-x402`` on PyPI (see the ``sincor-x402``
optional-dependencies extra in crewai-tools' pyproject).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Protocol, runtime_checkable
import urllib.error
import urllib.request


class SincorError(Exception):
    """Base error for the SINCOR client."""


class SincorAPIError(SincorError):
    """Error returned by (or while calling) the SINCOR platform API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SincorNotFound(SincorAPIError):
    """The requested task or bid does not exist."""


class TaskStatus(str, Enum):
    """Lifecycle status of a SINCOR task."""

    OPEN = "open"
    AUCTION_OPEN = "auction_open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    SETTLED = "settled"
    RELEASED = "released"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskKind(str, Enum):
    """Kind of SINCOR task."""

    ESCROW = "escrow"
    AUCTION = "auction"  # CNP (Contract Net Protocol) auction


@dataclass
class Bid:
    """A bid placed on a SINCOR auction task."""

    id: str
    task_id: str
    bidder_id: str
    amount: float
    currency: str = "USDC"
    message: str = ""
    eta_seconds: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: dict) -> "Bid":
        return cls(
            id=str(d.get("id") or d.get("bid_id")),
            task_id=str(d.get("task_id", "")),
            bidder_id=str(d.get("bidder_id") or d.get("agent_id", "")),
            amount=float(d.get("amount", 0)),
            currency=str(d.get("currency", "USDC")),
            message=str(d.get("message", "")),
            eta_seconds=d.get("eta_seconds"),
            raw=dict(d),
        )


@dataclass
class EscrowTask:
    """A SINCOR task (escrowed or auction)."""

    id: str
    title: str
    description: str = ""
    budget_amount: float = 0.0
    currency: str = "USDC"
    status: TaskStatus = TaskStatus.OPEN
    kind: TaskKind = TaskKind.ESCROW
    accepted_bid_id: str | None = None
    idempotency_key: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: dict) -> "EscrowTask":
        try:
            status = TaskStatus(d.get("status", "open"))
        except ValueError:
            status = TaskStatus.OPEN
        try:
            kind = TaskKind(d.get("kind", "escrow"))
        except ValueError:
            kind = TaskKind.ESCROW
        return cls(
            id=str(d.get("id") or d.get("task_id")),
            title=str(d.get("title", "")),
            description=str(d.get("description", "")),
            budget_amount=float(d.get("budget_amount") or d.get("amount") or 0),
            currency=str(d.get("currency", "USDC")),
            status=status,
            kind=kind,
            accepted_bid_id=d.get("accepted_bid_id"),
            idempotency_key=d.get("idempotency_key"),
            raw=dict(d),
        )


@dataclass
class SettlementRecord:
    """Settlement record for a task — the platform fee is explicit, never hidden."""

    task_id: str
    bid_id: str | None
    gross_amount: float
    platform_fee: float
    platform_fee_rate: float  # 0.05 = the 5% platform fee
    net_to_worker: float
    currency: str = "USDC"
    status: str = "settled"
    tx_hash: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: dict) -> "SettlementRecord":
        gross = float(d.get("gross_amount") or d.get("amount") or 0)
        fee = float(d.get("platform_fee", round(gross * 0.05, 6)))
        net = float(d.get("net_to_worker", round(gross - fee, 6)))
        return cls(
            task_id=str(d.get("task_id", "")),
            bid_id=d.get("bid_id"),
            gross_amount=gross,
            platform_fee=fee,
            platform_fee_rate=float(d.get("platform_fee_rate", 0.05)),
            net_to_worker=net,
            currency=str(d.get("currency", "USDC")),
            status=str(d.get("status", "settled")),
            tx_hash=d.get("tx_hash"),
            raw=dict(d),
        )


@runtime_checkable
class Transport(Protocol):
    """Transport protocol: one JSON request/response round trip."""

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Perform a request against the platform API and return parsed JSON."""
        ...


class HttpTransport:
    """HTTP transport based on `urllib` (no third-party dependencies)."""

    def __init__(
        self,
        base_url: str = "https://getsincor.com",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Perform an HTTP request and return the parsed JSON body."""
        url = f"{self.base_url}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        data = json.dumps(json_body).encode() if json_body is not None else None
        req = urllib.request.Request(url, data=data, method=method.upper())
        req.add_header("User-Agent", "Mozilla/5.0 crewai-tools-sincor/0.1.0")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise SincorNotFound(f"Not found: {path}", status_code=404) from e
            raise SincorAPIError(f"HTTP {e.code} on {path}", status_code=e.code) from e


class SincorClient:
    """Typed client for the SINCOR platform API."""

    def __init__(
        self,
        base_url: str = "https://getsincor.com",
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport: Transport = (
            transport
            if transport is not None
            else HttpTransport(base_url=base_url, api_key=api_key)
        )

    def create_escrowed_task(
        self,
        title: str,
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        *,
        idempotency_key: str | None = None,
    ) -> EscrowTask:
        """Create an escrowed task; replaying `idempotency_key` returns the same task."""
        body = {
            "title": title,
            "description": description,
            "budget_amount": budget_amount,
            "currency": currency,
            "idempotency_key": idempotency_key,
        }
        return EscrowTask.from_api(
            self.transport.request("POST", "/api/tasks", json_body=body)
        )

    def post_auction_task(
        self,
        title: str,
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        *,
        idempotency_key: str | None = None,
    ) -> EscrowTask:
        """Post a CNP (Contract Net Protocol) auction task."""
        body = {
            "title": title,
            "description": description,
            "budget_amount": budget_amount,
            "currency": currency,
            "idempotency_key": idempotency_key,
        }
        return EscrowTask.from_api(
            self.transport.request("POST", "/api/tasks/auction", json_body=body)
        )

    def list_bids(self, task_id: str) -> list[Bid]:
        """List bids on a task."""
        data = self.transport.request("GET", f"/api/tasks/{task_id}/bids")
        items = data.get("bids", data) if isinstance(data, dict) else data
        return [Bid.from_api(b) for b in items]

    def place_bid(
        self,
        task_id: str,
        amount: float,
        bidder_id: str,
        currency: str = "USDC",
        message: str = "",
    ) -> Bid:
        """Place a bid on an auction task."""
        data = self.transport.request(
            "POST",
            f"/api/tasks/{task_id}/bids",
            json_body={
                "amount": amount,
                "bidder_id": bidder_id,
                "currency": currency,
                "message": message,
            },
        )
        return Bid.from_api(data)

    def accept_bid(self, task_id: str, bid_id: str) -> EscrowTask:
        """Accept a bid, assigning the task to that bidder."""
        return EscrowTask.from_api(
            self.transport.request(
                "POST", f"/api/tasks/{task_id}/accept", json_body={"bid_id": bid_id}
            )
        )

    def check_status(self, task_id: str) -> EscrowTask:
        """Fetch the current state of a task."""
        return EscrowTask.from_api(self.transport.request("GET", f"/api/tasks/{task_id}"))

    def get_settlement(self, task_id: str) -> SettlementRecord:
        """Fetch the settlement record for a settled task."""
        return SettlementRecord.from_api(
            self.transport.request("GET", f"/api/tasks/{task_id}/settlement")
        )

    def release(self, task_id: str) -> SettlementRecord:
        """Release escrow to the accepted worker; returns the settlement record."""
        return SettlementRecord.from_api(
            self.transport.request("POST", f"/api/tasks/{task_id}/release", json_body={})
        )

    def settle(self, task_id: str) -> SettlementRecord:
        """Settle a task whose bid was accepted; returns the settlement record."""
        return SettlementRecord.from_api(
            self.transport.request("POST", f"/api/tasks/{task_id}/settle", json_body={})
        )

    def refund(self, task_id: str) -> EscrowTask:
        """Refund an escrowed task to its poster."""
        return EscrowTask.from_api(
            self.transport.request("POST", f"/api/tasks/{task_id}/refund", json_body={})
        )


__all__ = [
    "Bid",
    "EscrowTask",
    "HttpTransport",
    "SettlementRecord",
    "SincorAPIError",
    "SincorClient",
    "SincorError",
    "SincorNotFound",
    "TaskKind",
    "TaskStatus",
    "Transport",
]
