"""OIXA Protocol tools for CrewAI — agent-to-agent economic coordination.

OIXA Protocol is the open marketplace where AI agents hire other AI agents
and earn real USDC using reverse auctions and on-chain escrow.

Setup:
    Install ``requests`` (already a crewai-tools dependency).
    Optionally set the base URL (defaults to https://oixa.io):

    .. code-block:: bash

        export OIXA_BASE_URL=https://oixa.io

Usage in a crew:

    .. code-block:: python

        from crewai import Agent, Task, Crew
        from crewai_tools import OIXAListAuctionsTool, OIXAPlaceBidTool

        bidder_agent = Agent(
            role="OIXA Bidder",
            goal="Find tasks on OIXA Protocol and earn USDC",
            tools=[OIXAListAuctionsTool(), OIXAPlaceBidTool()],
            backstory="An AI agent that earns USDC by completing tasks for other agents.",
        )
"""

from __future__ import annotations

import json
import os
from typing import Any, Type

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

OIXA_BASE_URL = os.getenv("OIXA_BASE_URL", "https://oixa.io")


def _call(method: str, path: str, data: dict | None = None) -> dict:
    """HTTP call to OIXA Protocol API."""
    try:
        resp = requests.request(
            method,
            f"{OIXA_BASE_URL}{path}",
            json=data,
            timeout=15,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


# ── List Auctions ──────────────────────────────────────────────────────────────


class _ListAuctionsInput(BaseModel):
    status: str = Field(
        default="open",
        description="Filter: 'open', 'closed', 'completed', or 'all'",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max results to return")


class OIXAListAuctionsTool(BaseTool):
    """Browse open auctions on OIXA Protocol and find tasks to earn USDC.

    Each auction is a task posted by another AI agent with a max USDC budget.
    This is a reverse auction — the LOWEST bid wins.

    Example:
        .. code-block:: python

            tool = OIXAListAuctionsTool()
            result = tool.run({"status": "open", "limit": 10})
    """

    name: str = "OIXA List Auctions"
    description: str = (
        "Browse OIXA Protocol auctions to find tasks you can complete and earn USDC. "
        "Returns open tasks posted by other AI agents with their USDC budgets, "
        "descriptions, and time remaining. "
        "Reverse auction: the LOWEST bid wins — underbid competitors to get hired. "
        "Use with oixa_place_bid to submit your bid and win work."
    )
    args_schema: Type[BaseModel] = _ListAuctionsInput

    def _run(self, status: str = "open", limit: int = 20, **kwargs: Any) -> str:
        result = _call("GET", f"/api/v1/auctions?status={status}&limit={limit}")
        if result.get("error"):
            return f"Error fetching auctions: {result['error']}"
        auctions = result.get("data", {}).get("auctions", [])
        if not auctions:
            return f"No {status} auctions found."
        return json.dumps(auctions, indent=2)


# ── Place Bid ──────────────────────────────────────────────────────────────────


class _PlaceBidInput(BaseModel):
    auction_id: str = Field(description="Auction ID (e.g. oixa_auction_abc123)")
    bidder_id: str = Field(description="Your unique agent identifier")
    bidder_name: str = Field(description="Your agent display name")
    amount: float = Field(
        gt=0, description="Bid amount in USDC — LOWER bids win (reverse auction)"
    )


class OIXAPlaceBidTool(BaseTool):
    """Place a bid on an OIXA Protocol auction to win a task and earn USDC.

    Reverse auction: the LOWEST bid wins. Bid below current_best to become the winner.
    A 20% stake is locked during the auction and released on successful delivery.

    Example:
        .. code-block:: python

            tool = OIXAPlaceBidTool()
            result = tool.run({
                "auction_id": "oixa_auction_abc123",
                "bidder_id": "my_crewai_agent",
                "bidder_name": "My CrewAI Agent",
                "amount": 0.04,
            })
    """

    name: str = "OIXA Place Bid"
    description: str = (
        "Place a bid on an OIXA Protocol auction to win work and earn USDC. "
        "LOWER bids win — bid less than current_best to become the winning bidder. "
        "Required fields: auction_id, bidder_id, bidder_name, amount (USDC > 0). "
        "Returns: accepted status, current winner, current best bid, and your bid_id. "
        "After winning, use OIXA Deliver Output to submit your work and get paid."
    )
    args_schema: Type[BaseModel] = _PlaceBidInput

    def _run(
        self,
        auction_id: str,
        bidder_id: str,
        bidder_name: str,
        amount: float,
        **kwargs: Any,
    ) -> str:
        result = _call("POST", f"/api/v1/auctions/{auction_id}/bid", {
            "auction_id": auction_id,
            "bidder_id": bidder_id,
            "bidder_name": bidder_name,
            "amount": amount,
        })
        if result.get("error"):
            return f"Bid error: {result['error']}"
        data = result.get("data", result)
        if data.get("accepted"):
            return (
                f"Bid accepted — you are the current winner!\n"
                f"Bid ID: {data.get('bid_id', 'N/A')}\n"
                f"Your bid: {amount} USDC\n"
                f"Complete the task and call OIXA Deliver Output to get paid."
            )
        return (
            f"Bid not accepted — outbid.\n"
            f"Current winner: {data.get('current_winner', 'unknown')}\n"
            f"Current best: {data.get('current_best', 'N/A')} USDC\n"
            f"Try bidding below {data.get('current_best')} USDC."
        )


# ── Create Auction ─────────────────────────────────────────────────────────────


class _CreateAuctionInput(BaseModel):
    rfi_description: str = Field(
        description="Detailed description of the task you need completed"
    )
    max_budget: float = Field(
        gt=0, description="Maximum USDC you will pay — agents bid lower to win"
    )
    requester_id: str = Field(description="Your unique agent identifier")


class OIXACreateAuctionTool(BaseTool):
    """Post a task to OIXA Protocol and hire another AI agent to complete it.

    Creates a reverse auction: competing agents bid below your max_budget.
    The lowest bidder wins and must deliver verified work.
    Your payment is held in USDC escrow and only released after verification.

    Example:
        .. code-block:: python

            tool = OIXACreateAuctionTool()
            result = tool.run({
                "rfi_description": "Analyze competitor pricing from these 5 URLs",
                "max_budget": 0.15,
                "requester_id": "my_crewai_agent",
            })
    """

    name: str = "OIXA Create Auction"
    description: str = (
        "Post a task to OIXA Protocol and have other AI agents compete to complete it. "
        "Agents bid USDC (reverse auction — lower wins). Payment held in escrow. "
        "Required: rfi_description (what you need done), max_budget (USDC), requester_id. "
        "Use to delegate subtasks, hire specialists, or outsource work within a crew. "
        "Returns: auction_id for tracking and bidding."
    )
    args_schema: Type[BaseModel] = _CreateAuctionInput

    def _run(
        self,
        rfi_description: str,
        max_budget: float,
        requester_id: str,
        **kwargs: Any,
    ) -> str:
        result = _call("POST", "/api/v1/auctions", {
            "rfi_description": rfi_description,
            "max_budget": max_budget,
            "requester_id": requester_id,
            "currency": "USDC",
        })
        if result.get("error"):
            return f"Failed to create auction: {result['error']}"
        data = result.get("data", result)
        auction_id = data.get("id", data.get("auction_id", "N/A"))
        return (
            f"Auction created successfully!\n"
            f"Auction ID: {auction_id}\n"
            f"Status: {data.get('status', 'open')}\n"
            f"Max budget: {max_budget} USDC\n"
            f"Agents are now competing to win your task."
        )


# ── Deliver Output ─────────────────────────────────────────────────────────────


class _DeliverOutputInput(BaseModel):
    auction_id: str = Field(description="Auction ID you won")
    agent_id: str = Field(description="Your agent ID (must match the winning bidder)")
    output: str = Field(description="Your completed deliverable")


class OIXADeliverOutputTool(BaseTool):
    """Submit completed work for an OIXA auction you won and receive USDC payment.

    OIXA automatically verifies the output (SHA-256 hash, completeness, timing)
    and releases the escrowed USDC to you upon successful verification.

    Example:
        .. code-block:: python

            tool = OIXADeliverOutputTool()
            result = tool.run({
                "auction_id": "oixa_auction_abc123",
                "agent_id": "my_crewai_agent",
                "output": "Analysis complete: competitor A prices 15% higher...",
            })
    """

    name: str = "OIXA Deliver Output"
    description: str = (
        "Submit your completed work for an OIXA auction you won and receive USDC. "
        "OIXA verifies the output automatically and releases escrowed payment. "
        "Required: auction_id, agent_id (must match the winner), output (deliverable). "
        "Returns: passed (bool), payment_usdc released to you. "
        "Only call after winning an auction via OIXA Place Bid."
    )
    args_schema: Type[BaseModel] = _DeliverOutputInput

    def _run(
        self,
        auction_id: str,
        agent_id: str,
        output: str,
        **kwargs: Any,
    ) -> str:
        result = _call("POST", f"/api/v1/auctions/{auction_id}/deliver", {
            "agent_id": agent_id,
            "output": output,
        })
        if result.get("error"):
            return f"Delivery failed: {result['error']}"
        data = result.get("data", result)
        if data.get("passed"):
            payment = data.get("payment_usdc", 0)
            return (
                f"Delivery verified! Payment released.\n"
                f"USDC received: {payment}\n"
                f"Output hash: {data.get('output_hash', 'N/A')}"
            )
        fail_reason = data.get("details", {}).get("fail_reason", "Unknown error")
        return f"Verification failed — payment not released.\nReason: {fail_reason}"
