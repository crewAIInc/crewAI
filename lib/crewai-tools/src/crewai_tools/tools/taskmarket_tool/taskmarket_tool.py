"""Read-only CrewAI tools for discovering work on TaskMarket.

These tools intentionally expose only public GET endpoints. They do not create
tasks, submit work, sign messages, access wallets, or move funds. A host
application must implement its own approval and wallet boundary for any
write-side TaskMarket action.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TaskMarketSearchInput(BaseModel):
    """Arguments for a bounded, read-only TaskMarket search."""

    max_reward_usdc: Decimal = Field(
        Decimal("1"),
        ge=Decimal("0"),
        le=Decimal("1000000"),
        description="Only return open tasks at or below this USDC reward ceiling.",
    )
    tags: str = Field(
        "",
        description="Optional comma-separated tags to pass to TaskMarket.",
    )
    limit: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of candidate tasks to inspect.",
    )


class TaskMarketGetTaskInput(BaseModel):
    """Arguments for reading one public TaskMarket task."""

    task_id: str = Field(..., min_length=1, description="Opaque TaskMarket task ID.")


class _TaskMarketReadOnlyTool(BaseTool):
    """Shared HTTP and reward parsing helpers for the public tools."""

    base_url: str = "https://api.taskmarket.dev/api"
    request_timeout: int = Field(default=20, ge=1, le=120)
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.taskmarket.dev/api"

    @staticmethod
    def _reward_usdc(row: Mapping[str, Any]) -> Decimal:
        """Normalize a TaskMarket reward into decimal USDC units."""
        raw = row.get("rewardUsdc", row.get("reward", "0"))
        try:
            value = Decimal(str(raw))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"TaskMarket returned an invalid reward: {raw!r}") from exc

        # TaskMarket's canonical reward is an integer in six-decimal USDC base
        # units. Cached/page payloads may provide rewardUsdc as a decimal.
        if "rewardUsdc" not in row and (isinstance(raw, int) or "." not in str(raw)):
            value /= Decimal("1000000")
        return value

    def _get_json(self, path: str) -> Any:
        """Fetch and decode one public TaskMarket JSON endpoint."""
        request = Request(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"TaskMarket HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"TaskMarket network error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("TaskMarket returned non-JSON data") from exc

    @staticmethod
    def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
        """Return the stable, non-sensitive fields exposed to an agent."""
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return {
            "id": str(row.get("id", "")),
            "reward_usdc": str(_TaskMarketReadOnlyTool._reward_usdc(row)),
            "status": str(row.get("status", "")),
            "mode": str(row.get("mode", row.get("taskMode", ""))),
            "description": str(row.get("description", "")),
            "deadline": row.get("deadline") or row.get("expiryTime"),
            "tags": [str(tag) for tag in tags],
        }


class TaskMarketSearchTool(_TaskMarketReadOnlyTool):
    """Find open TaskMarket jobs without touching a wallet or signing anything."""

    name: str = "taskmarket_search_open_work"
    description: str = (
        "Read-only discovery of public TaskMarket jobs. Returns open tasks at "
        "or below a USDC reward ceiling; never submits work or moves funds."
    )
    args_schema: type[BaseModel] = TaskMarketSearchInput

    def _run(
        self,
        max_reward_usdc: Decimal = Decimal("1"),
        tags: str = "",
        limit: int = 20,
    ) -> str:
        """Return qualifying open tasks without performing a write operation."""
        args = TaskMarketSearchInput(
            max_reward_usdc=max_reward_usdc,
            tags=tags,
            limit=limit,
        )
        params: dict[str, str] = {
            "status": "open",
            "sort": "reward_asc",
            "limit": str(args.limit),
        }
        if args.tags.strip():
            params["tags"] = ",".join(
                tag.strip() for tag in args.tags.split(",") if tag.strip()
            )
        payload = self._get_json(f"tasks?{urlencode(params)}")
        rows = payload.get("tasks", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise RuntimeError("TaskMarket returned an unexpected task-list shape")

        results = []
        for row in rows[: args.limit]:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("status", "")).lower() != "open":
                continue
            if self._reward_usdc(row) <= args.max_reward_usdc:
                results.append(self._public_row(row))
        return json.dumps(results, ensure_ascii=False)


class TaskMarketGetTaskTool(_TaskMarketReadOnlyTool):
    """Read one public TaskMarket task by its opaque ID."""

    name: str = "taskmarket_get_task"
    description: str = (
        "Read one public TaskMarket task by opaque ID. This is read-only and "
        "does not claim, submit, sign, or pay for anything."
    )
    args_schema: type[BaseModel] = TaskMarketGetTaskInput

    def _run(self, task_id: str) -> str:
        """Return one public task after validating and encoding its opaque ID."""
        if not task_id or task_id in {".", ".."} or "/" in task_id or "\\" in task_id:
            raise ValueError("task_id must be a non-empty opaque ID")
        payload = self._get_json(f"tasks/{quote(task_id, safe='')}")
        if not isinstance(payload, Mapping):
            raise RuntimeError("TaskMarket returned an unexpected task shape")
        return json.dumps(self._public_row(payload), ensure_ascii=False)
