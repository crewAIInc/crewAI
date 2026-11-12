"""Tests for the SINCOR settlement tools.

All tests run against a MockTransport — an in-memory simulation of the
SINCOR platform API. No network access is used.
"""

import json
import uuid

from crewai_tools import SincorAuctionTool, SincorEscrowTool, SincorSettlementTool
from crewai_tools.tools.sincor_settlement_tool.sincor_client import (
    SincorAPIError,
    SincorClient,
    SincorNotFound,
)


class MockTransport:
    """In-memory mock of the SINCOR platform API (5% fee on settle/release)."""

    PLATFORM_FEE_RATE = 0.05

    def __init__(self):
        self.tasks = {}
        self.bids = {}
        self.settlements = {}
        self._idem = {}
        self._counter = 0

    def _next_id(self, prefix):
        self._counter += 1
        return f"{prefix}_{self._counter:04d}_{uuid.uuid4().hex[:6]}"

    def add_bid(self, task_id, *, bidder_id="worker-agent-01", amount, currency="USDC",
                message="", eta_seconds=None):
        if task_id not in self.tasks:
            raise SincorNotFound(f"task {task_id} not found")
        bid = {"id": self._next_id("bid"), "task_id": task_id, "bidder_id": bidder_id,
               "amount": float(amount), "currency": currency, "message": message,
               "eta_seconds": eta_seconds}
        self.bids.setdefault(task_id, []).append(bid)
        return bid

    def request(self, method, path, *, json_body=None, params=None):
        method = method.upper()
        body = json_body or {}
        parts = [p for p in path.split("/") if p]
        if method == "POST" and path in ("/api/tasks", "/api/tasks/"):
            return self._create_task(body, kind="escrow")
        if method == "POST" and path == "/api/tasks/auction":
            return self._create_task(body, kind="auction")
        if method == "GET" and len(parts) == 3 and parts[:2] == ["api", "tasks"]:
            tid = parts[2]
            if tid not in self.tasks:
                raise SincorNotFound(f"task {tid} not found")
            return dict(self.tasks[tid])
        if method == "GET" and len(parts) == 4 and parts[3] == "bids":
            tid = parts[2]
            if tid not in self.tasks:
                raise SincorNotFound(f"task {tid} not found")
            return {"bids": [dict(b) for b in self.bids.get(tid, [])]}
        if method == "POST" and len(parts) == 4 and parts[3] == "bids":
            return self.add_bid(parts[2], bidder_id=body.get("bidder_id", "worker-agent-01"),
                                amount=float(body["amount"]), currency=body.get("currency", "USDC"),
                                message=body.get("message", ""), eta_seconds=body.get("eta_seconds"))
        if method == "POST" and len(parts) == 4 and parts[3] == "accept":
            tid = parts[2]
            if tid not in self.tasks:
                raise SincorNotFound(f"task {tid} not found")
            bid_id = body.get("bid_id")
            bid = next((b for b in self.bids.get(tid, []) if b["id"] == bid_id), None)
            if bid is None:
                raise SincorNotFound(f"bid {bid_id} not found on task {tid}")
            self.tasks[tid]["status"] = "assigned"
            self.tasks[tid]["accepted_bid_id"] = bid_id
            return dict(self.tasks[tid])
        if method == "POST" and len(parts) == 4 and parts[3] in ("release", "settle"):
            tid = parts[2]
            if tid not in self.tasks:
                raise SincorNotFound(f"task {tid} not found")
            task = self.tasks[tid]
            bid_id = task.get("accepted_bid_id")
            bid = next((b for b in self.bids.get(tid, []) if b["id"] == bid_id), None)
            gross = float(bid["amount"]) if bid else float(task.get("budget_amount", 0))
            fee = round(gross * self.PLATFORM_FEE_RATE, 6)
            settlement = {"task_id": tid, "bid_id": bid_id, "gross_amount": gross,
                          "platform_fee": fee, "platform_fee_rate": self.PLATFORM_FEE_RATE,
                          "net_to_worker": round(gross - fee, 6),
                          "currency": task.get("currency", "USDC"),
                          "status": "settled", "tx_hash": None}
            self.settlements[tid] = settlement
            task["status"] = "settled" if parts[3] == "settle" else "released"
            return dict(settlement)
        if method == "POST" and len(parts) == 4 and parts[3] == "refund":
            tid = parts[2]
            if tid not in self.tasks:
                raise SincorNotFound(f"task {tid} not found")
            self.tasks[tid]["status"] = "refunded"
            return dict(self.tasks[tid])
        if method == "GET" and len(parts) == 4 and parts[3] == "settlement":
            tid = parts[2]
            if tid not in self.settlements:
                raise SincorNotFound(f"no settlement for {tid}")
            return dict(self.settlements[tid])
        raise SincorAPIError(f"MockTransport: unhandled {method} {path}")

    def _create_task(self, body, *, kind):
        idem = body.get("idempotency_key")
        if idem and idem in self._idem:
            return dict(self.tasks[self._idem[idem]])
        tid = self._next_id("tsk")
        task = {"id": tid, "title": body.get("title", ""), "description": body.get("description", ""),
                "budget_amount": float(body.get("budget_amount") or body.get("amount") or 0),
                "currency": body.get("currency", "USDC"),
                "status": "auction_open" if kind == "auction" else "open",
                "kind": kind, "accepted_bid_id": None, "idempotency_key": idem}
        self.tasks[tid] = task
        self.bids[tid] = []
        if idem:
            self._idem[idem] = tid
        return dict(task)


def test_sincor_auction_full_loop():
    mock = MockTransport()
    tool = SincorSettlementTool(client=SincorClient(transport=mock))
    posted = json.loads(tool._run(action="post", title="Crew job", budget_amount=100.0))
    assert posted["status"] == "auction_open"
    mock.add_bid(posted["task_id"], amount=100.0, bidder_id="crew-worker")
    bids = json.loads(tool._run(action="list_bids", task_id=posted["task_id"]))["bids"]
    assert len(bids) == 1
    accepted = json.loads(tool._run(action="accept", task_id=posted["task_id"], bid_id=bids[0]["bid_id"]))
    assert accepted["status"] == "assigned"
    settled = json.loads(tool._run(action="settle", task_id=posted["task_id"]))
    assert settled["gross_amount"] == 100.0
    assert settled["platform_fee"] == 5.0
    assert settled["platform_fee_rate"] == 0.05
    assert settled["net_to_worker"] == 95.0


def test_sincor_escrow_tool():
    mock = MockTransport()
    tool = SincorEscrowTool(client=SincorClient(transport=mock))
    created = json.loads(tool._run(action="create", title="E", budget_amount=20.0))
    assert created["status"] == "open"
    refunded = json.loads(tool._run(action="refund", task_id=created["task_id"]))
    assert refunded["status"] == "refunded"


def test_sincor_alias_is_auction_subclass():
    assert issubclass(SincorSettlementTool, SincorAuctionTool)
