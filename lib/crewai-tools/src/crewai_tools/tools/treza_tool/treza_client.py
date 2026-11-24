"""A small client for Treza's REST API (POST /api/v1/tools/<name>).

Every render follows the same order: ask Treza for a quote, stop if the quote
is over the max cost the user set, start the render once at the quoted price,
then wait for the finished file.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any
import uuid

import requests


API_URL = "https://www.trezalabs.com"
KEYS_PAGE = "https://www.trezalabs.com/platform/api-keys"
TOP_UP_URL = "https://www.trezalabs.com/platform/settings"
USER_AGENT = "crewai-tools-treza"
FINISHED = {"success", "error", "partial", "cancelled"}
STALE_QUOTE_CODES = {"STALE_QUOTE", "QUOTE_MISMATCH", "UNKNOWN_QUOTE"}

Data = dict[str, Any]


class TrezaError(Exception):
    """A refusal from Treza, worded so a person or a model can act on it."""

    def __init__(self, message: str, status: int = 0, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


def compact(fields: Data) -> Data:
    """Drop empty values, so an optional field left out gets Treza's default."""
    return {k: v for k, v in fields.items() if v is not None and v != "" and v != []}


def to_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


class TrezaClient:
    def __init__(self, api_key: str, session: requests.Session | None = None):
        self.api_key = (api_key or "").strip()
        self.session = session or requests.Session()

    def request(self, tool: str, args: Data | None = None) -> tuple[bool, int, Data]:
        """Call one Treza tool without raising on a refusal: (ok, HTTP status, body)."""
        response = self.session.post(
            f"{API_URL}/api/v1/tools/{tool}",
            json=compact(args or {}),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
            },
            timeout=(10, 60),
        )
        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text[:300]}
        if not isinstance(body, dict):
            body = {"result": body}
        return response.status_code < 400, response.status_code, body

    def call(self, tool: str, args: Data | None = None) -> Data:
        ok, status, body = self.request(tool, args)
        if ok:
            return body
        raise refusal(status, body)

    def start_quoted(
        self, max_cost_usd: float, quote: Callable[[], tuple[bool, int, Data]]
    ) -> Data:
        """Quote, check the cap and the balance, then start the render once."""
        for attempt in range(3):
            ok, status, quoted = quote()
            if not ok:
                if quoted.get("code") == "OVER_BUDGET":
                    raise over_budget(
                        to_number(quoted.get("estimatedChargeUsd")), max_cost_usd
                    )
                raise refusal(status, quoted)
            pipeline_id = str(quoted.get("pipelineId"))
            price = to_number(quoted.get("estimatedChargeUsd"))
            if price is None or price > max_cost_usd:
                raise over_budget(price, max_cost_usd)
            if quoted.get("sufficient") is False:
                balance = to_number(quoted.get("balanceUsd")) or 0
                raise TrezaError(
                    f"This render costs ${price:.2f} and the Treza balance is "
                    f"${balance:.2f}. Nothing was spent. Add credits at "
                    f"{quoted.get('topUpUrl') or TOP_UP_URL}",
                    402,
                    "InsufficientCredits",
                )
            quote_id = quoted.get("quoteId")
            start: Data = {"pipelineId": pipeline_id}
            if isinstance(quote_id, str):
                start["quoteId"] = quote_id
            else:
                start["requestId"] = str(uuid.uuid4())
            ok, status, run = self.request("run_pipeline", start)
            if ok:
                return {
                    "quoted": quoted,
                    "runId": str(run.get("runId")),
                    "pipelineId": pipeline_id,
                    "priceUsd": price,
                }
            code = run.get("code")
            if attempt < 2 and code in STALE_QUOTE_CODES:
                continue
            if code == "PRICE_CHANGED":
                new_price = to_number(run.get("estimatedChargeUsd"))
                if new_price is None or new_price > max_cost_usd:
                    raise over_budget(new_price, max_cost_usd)
                again = self.call(
                    "run_pipeline",
                    {"pipelineId": pipeline_id, "requestId": str(uuid.uuid4())},
                )
                return {
                    "quoted": quoted,
                    "runId": str(again.get("runId")),
                    "pipelineId": pipeline_id,
                    "priceUsd": new_price,
                }
            raise refusal(status, run)
        raise TrezaError("Treza's quote kept changing. Try again.")

    def wait_for_run(self, pipeline_id: str, run_id: str, wait_seconds: float) -> Data:
        """Poll a run until it finishes or the wait is used up. Returns the run as last seen."""
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            run = self.call("get_run", {"pipelineId": pipeline_id, "runId": run_id})
            if run.get("status") != "running":
                return run
            pause = min(max(to_number(run.get("pollAfterSec")) or 3, 2), 10)
            if time.monotonic() + pause >= deadline:
                return run
            time.sleep(pause)


def refusal(status: int, body: Data) -> TrezaError:
    error = body.get("error")
    message = (
        error if isinstance(error, str) and error else f"Treza answered HTTP {status}."
    )
    code = body.get("code") if isinstance(body.get("code"), str) else None
    if status == 401:
        return TrezaError(
            f"{message} Check the Treza API key ({KEYS_PAGE}).", status, code
        )
    if status == 402:
        return TrezaError(
            f"{message} Add credits at {body.get('topUpUrl') or TOP_UP_URL}",
            status,
            code or "InsufficientCredits",
        )
    if status == 429:
        wait = int(to_number(body.get("retryAfterSec")) or 60)
        return TrezaError(f"{message} Wait {wait} seconds and try again.", status, code)
    return TrezaError(str(message), status, code)


def over_budget(price: float | None, max_cost_usd: float) -> TrezaError:
    shown = f"${price:.2f}" if price is not None else "more"
    return TrezaError(
        f"This render costs {shown}, over the max cost of ${max_cost_usd:.2f}. "
        "Nothing was spent. Raise max_cost_usd, or ask for less (a shorter clip "
        "or a cheaper model).",
        400,
        "OverBudget",
    )


def primary_url(run: Data, key: str | None = None) -> str | None:
    """The finished file: the named output, or else the first URL among the outputs."""
    raw = run.get("outputs")
    outputs: Data = raw if isinstance(raw, dict) else {}
    named = outputs.get(key) if key else None
    if isinstance(named, str):
        return named
    for value in outputs.values():
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def finished_or_raise(run: Data) -> None:
    """A render that broke or was cancelled raises, with the reason Treza gave."""
    status = run.get("status")
    if status not in ("error", "cancelled"):
        return
    reasons = [run.get("statusNote"), run.get("refusalNote"), run.get("cancelReason")]
    for node in run.get("nodes") or []:
        if isinstance(node, dict) and isinstance(node.get("error"), str):
            reasons.append(f"{node.get('label') or node.get('type')}: {node['error']}")
            break
    said = " ".join(r for r in reasons if isinstance(r, str) and r)
    raise TrezaError(
        f"The render stopped before it finished. {said}".strip(),
        0,
        "RenderCancelled" if status == "cancelled" else "RenderStopped",
    )


def run_item(
    run: Data, output_key: str | None = None, extra: Data | None = None
) -> Data:
    """One render as a flat result: the file first, then status, price and the IDs to look it up."""
    status = str(run.get("status") or "running")
    duration_ms = to_number(run.get("durationMs"))
    return compact(
        {
            "url": primary_url(run, output_key),
            "status": status,
            "finished": status in FINISHED,
            **(extra or {}),
            "render_seconds": round(duration_ms / 1000, 1)
            if duration_ms is not None
            else None,
            "run_id": run.get("runId"),
            "pipeline_id": run.get("pipelineId"),
            "status_note": run.get("statusNote") or run.get("refusalNote"),
        }
    )
