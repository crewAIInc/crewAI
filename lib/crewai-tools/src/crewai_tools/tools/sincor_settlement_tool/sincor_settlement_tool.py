"""SINCOR escrow + CNP auction settlement tools for CrewAI.

Crews delegate work to other agents with no native way to bid the work
out, escrow the payment, or see what the platform took. These tools give
a crew a competitive CNP (Contract Net Protocol) auction before the work,
escrow holding during it, and a settlement record after it — gross, the
5% platform fee, and net to worker, all explicit — instead of a single
opaque paid call.

`SincorSettlementTool` is the headline import (an alias of
`SincorAuctionTool`).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from crewai_tools.tools.sincor_settlement_tool.sincor_client import SincorClient


class SincorEscrowToolSchema(BaseModel):
    """Input schema for `SincorEscrowTool`."""

    action: str = Field(
        default="create", description="create | status | release | refund | settlement"
    )
    title: str = Field(default="", description="Task title (create only)")
    description: str = Field(default="", description="Task description (create only)")
    budget_amount: float = Field(default=0.0, description="Escrow budget, e.g. in USDC")
    currency: str = Field(default="USDC")
    task_id: str = Field(
        default="", description="Task id (status/release/refund/settlement)"
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Idempotency key for create — the same key returns the same task",
    )


class SincorAuctionToolSchema(BaseModel):
    """Input schema for `SincorAuctionTool` / `SincorSettlementTool`."""

    action: str = Field(
        default="post", description="post | list_bids | accept | settle | status"
    )
    title: str = Field(default="", description="Task title (post only)")
    description: str = Field(default="", description="Task description (post only)")
    budget_amount: float = Field(
        default=0.0, description="Auction budget ceiling, e.g. in USDC"
    )
    currency: str = Field(default="USDC")
    task_id: str = Field(default="", description="Task id (list_bids/accept/settle/status)")
    bid_id: str = Field(default="", description="Bid id (accept only)")


class SincorEscrowTool(BaseTool):
    """Create and manage escrowed SINCOR tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Sincor Escrow"
    description: str = (
        "Create and manage escrowed SINCOR tasks. Release returns a settlement "
        "record showing the 5% platform fee and net to worker."
    )
    args_schema: Type[BaseModel] = SincorEscrowToolSchema
    client: Any = Field(default=None, exclude=True)

    def __init__(self, client: SincorClient | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "client", client if client is not None else SincorClient())

    def _run(
        self,
        action: str = "create",
        title: str = "",
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        task_id: str = "",
        idempotency_key: str | None = None,
        **kw: Any,
    ) -> str:
        c: SincorClient = self.client
        if action == "create":
            t = c.create_escrowed_task(
                title, description, budget_amount, currency, idempotency_key=idempotency_key
            )
            return json.dumps(
                {
                    "task_id": t.id,
                    "status": t.status.value,
                    "budget_amount": t.budget_amount,
                    "currency": t.currency,
                }
            )
        if action == "status":
            t = c.check_status(task_id)
            return json.dumps(
                {"task_id": t.id, "status": t.status.value, "accepted_bid_id": t.accepted_bid_id}
            )
        if action == "release":
            s = c.release(task_id)
            return json.dumps(
                {
                    "task_id": s.task_id,
                    "gross_amount": s.gross_amount,
                    "platform_fee": s.platform_fee,
                    "platform_fee_rate": s.platform_fee_rate,
                    "net_to_worker": s.net_to_worker,
                    "status": s.status,
                }
            )
        if action == "refund":
            t = c.refund(task_id)
            return json.dumps({"task_id": t.id, "status": t.status.value})
        if action == "settlement":
            s = c.get_settlement(task_id)
            return json.dumps(
                {
                    "task_id": s.task_id,
                    "gross_amount": s.gross_amount,
                    "platform_fee": s.platform_fee,
                    "net_to_worker": s.net_to_worker,
                }
            )
        raise ValueError(f"unknown action {action!r}")


class SincorAuctionTool(BaseTool):
    """Post CNP auction tasks on SINCOR, accept bids, and settle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Sincor Auction"
    description: str = (
        "Post a CNP auction task on SINCOR, list and accept bids, and settle. "
        "Settlement shows gross, 5% platform fee, and net."
    )
    args_schema: Type[BaseModel] = SincorAuctionToolSchema
    client: Any = Field(default=None, exclude=True)

    def __init__(self, client: SincorClient | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "client", client if client is not None else SincorClient())

    def _run(
        self,
        action: str = "post",
        title: str = "",
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        task_id: str = "",
        bid_id: str = "",
        **kw: Any,
    ) -> str:
        c: SincorClient = self.client
        if action == "post":
            t = c.post_auction_task(title, description, budget_amount, currency)
            return json.dumps(
                {"task_id": t.id, "status": t.status.value, "budget_amount": t.budget_amount}
            )
        if action == "list_bids":
            bids = c.list_bids(task_id)
            return json.dumps(
                {
                    "task_id": task_id,
                    "bids": [
                        {
                            "bid_id": b.id,
                            "bidder_id": b.bidder_id,
                            "amount": b.amount,
                            "currency": b.currency,
                        }
                        for b in bids
                    ],
                }
            )
        if action == "accept":
            t = c.accept_bid(task_id, bid_id)
            return json.dumps(
                {"task_id": t.id, "status": t.status.value, "accepted_bid_id": t.accepted_bid_id}
            )
        if action in ("settle", "release"):
            s = c.settle(task_id)
            return json.dumps(
                {
                    "task_id": s.task_id,
                    "bid_id": s.bid_id,
                    "gross_amount": s.gross_amount,
                    "platform_fee": s.platform_fee,
                    "platform_fee_rate": s.platform_fee_rate,
                    "net_to_worker": s.net_to_worker,
                    "status": s.status,
                }
            )
        if action == "status":
            t = c.check_status(task_id)
            return json.dumps({"task_id": t.id, "status": t.status.value})
        raise ValueError(f"unknown action {action!r}")


class SincorSettlementTool(SincorAuctionTool):
    """Escrowed CNP settlement under the headline name crew authors import.

    Alias of `SincorAuctionTool`: `from crewai_tools import SincorSettlementTool`.
    """

    name: str = "Sincor Settlement"
    description: str = (
        "SINCOR escrowed CNP settlement: post an auction task, accept a bid, "
        "settle with the 5% platform fee visible. Alias of SincorAuctionTool."
    )
