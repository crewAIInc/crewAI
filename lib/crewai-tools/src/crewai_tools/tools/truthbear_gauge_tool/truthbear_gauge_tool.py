"""Truth Bear GAUGE tools for CrewAI.

Official-source signal records (US federal agencies and named registries) with a
canonical record_hash the caller can recompute offline.

Design notes for reviewers:
  * Standard library only - this adds no dependency to crewai-tools.
  * No API key, no account, no signup. The two tools here use the free tier.
  * Neither tool holds or spends funds. The paid record endpoint answers HTTP 402
    with an x402 challenge; that challenge is handed back to the caller untouched
    and settling it is the caller's own wallet's job.
  * The catalog endpoint returns ~1.5 MB unfiltered, so a filter is required here
    rather than optional - an unfiltered dump would flood an agent's context.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import ClassVar, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, model_validator

BASE_URL = "https://aeml-x402.zeabur.app"
REQUEST_TIMEOUT = 20


def _get_json(path: str, params: Optional[dict] = None) -> dict:
    url = f"{BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body[:2000]}
        # In x402 a 402 is not a failure, it is the price quote. Return it intact.
        return {"http_status": exc.code, **payload}


class TruthBearCoverageInput(BaseModel):
    signal_id: Optional[str] = Field(
        None,
        description=(
            "Optional. A signal id in genus.species ASCII form, e.g. "
            "'hydrology.river-level', 'airquality.aqi', 'macro.unemployment'. "
            "Omit to get the summary across all 185 signal lines."
        ),
    )


class TruthBearCoverageTool(BaseTool):
    """Free: does this signal exist, how many objects, how fresh, is it on schedule."""

    name: str = "Truth Bear GAUGE coverage check"
    description: str = (
        "Free check, no key: for a signal line, report whether it exists, how many monitored "
        "objects it covers, how many of those readings are fresh / recent / stale, whether the "
        "source agency is on schedule or overdue, and the latest observation time. Use this "
        "before paying for a record, to confirm the line is covered and current. Sources are "
        "official government records and named registries (USGS, NOAA, NWS, EPA, SEC EDGAR, "
        "openFDA, EIA, BLS and others). An unknown signal id comes back with an empty signals "
        "list rather than an error."
    )
    args_schema: Type[BaseModel] = TruthBearCoverageInput
    package_dependencies: List[str] = []
    env_vars: List[EnvVar] = []

    def _run(self, signal_id: Optional[str] = None) -> str:
        params = {"summary": "1"}
        if signal_id:
            params["signal_id"] = signal_id
        data = _get_json("/gauge/coverage", params)
        signals = data.get("signals") or []
        if signal_id and not signals:
            return json.dumps(
                {
                    "found": False,
                    "signal_id": signal_id,
                    "hint": "List valid signal ids with the Truth Bear GAUGE catalog tool.",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"found": True, "totals": data.get("totals"), "signals": signals},
            ensure_ascii=False,
        )


class TruthBearCatalogInput(BaseModel):
    signal_id: Optional[str] = Field(
        None, description="Filter to one signal line, e.g. 'hydrology.river-level'."
    )
    industry: Optional[str] = Field(
        None, description="Filter to one industry, e.g. 'hydrology', 'airquality', 'macro'."
    )

    @model_validator(mode="after")
    def _require_a_filter(self):
        # The unfiltered catalog is ~1.5 MB. Returning that to an agent would blow its
        # context window, so a filter is required rather than merely recommended.
        if not self.signal_id and not self.industry:
            raise ValueError("Provide either signal_id or industry - the unfiltered catalog is ~1.5 MB.")
        return self


class TruthBearCatalogTool(BaseTool):
    """Free: which entity ids are valid for a signal line."""

    name: str = "Truth Bear GAUGE catalog lookup"
    description: str = (
        "Free lookup, no key: list the monitored object ids (entities) that are valid for a "
        "given signal line or industry, with their human-readable names. signal_id and entity "
        "must be used as a pair, so call this to get a valid entity before requesting a record. "
        "One of signal_id or industry is required."
    )
    args_schema: Type[BaseModel] = TruthBearCatalogInput
    package_dependencies: List[str] = []
    env_vars: List[EnvVar] = []

    def _run(self, signal_id: Optional[str] = None, industry: Optional[str] = None) -> str:
        args = TruthBearCatalogInput(signal_id=signal_id, industry=industry)
        params = {}
        if args.signal_id:
            params["signal_id"] = args.signal_id
        if args.industry:
            params["industry"] = args.industry
        return json.dumps(_get_json("/gauge/catalog", params), ensure_ascii=False)


class TruthBearRecordInput(BaseModel):
    signal_id: str = Field(..., description="Signal id, e.g. 'hydrology.river-level'.")
    entity: str = Field(
        ...,
        description="Monitored object id PAIRED WITH signal_id, e.g. '06893000'. Get valid "
        "pairs from the Truth Bear GAUGE catalog tool.",
    )


class TruthBearRecordTool(BaseTool):
    """Paid record - returns the x402 payment challenge, never settles it."""

    name: str = "Truth Bear GAUGE official record (paid, x402)"
    description: str = (
        "Request one official-source record for a signal_id and entity pair: the current reading, "
        "the official threshold band published by the source agency, a same-season percentile, and "
        "a canonical sha256 record_hash that can be recomputed offline so the answer does not have "
        "to be taken on trust. This endpoint is paid: it answers HTTP 402 with an x402 payment "
        "challenge stating network, asset, amount and recipient. THIS TOOL DOES NOT PAY. It returns "
        "the challenge unchanged so an x402-capable wallet in your own stack can settle it and "
        "retry. Charging is bound to HTTP 200 - if there is no data you are not billed."
    )
    args_schema: Type[BaseModel] = TruthBearRecordInput
    package_dependencies: List[str] = []
    env_vars: List[EnvVar] = []

    def _run(self, signal_id: str, entity: str) -> str:
        args = TruthBearRecordInput(signal_id=signal_id, entity=entity)
        data = _get_json("/gauge", {"signal_id": args.signal_id, "entity": args.entity})
        if data.get("http_status") == 402:
            return json.dumps(
                {
                    "payment_required": True,
                    "note": "Settle this x402 challenge with your own wallet, then retry the same URL.",
                    "challenge": {k: v for k, v in data.items() if k != "http_status"},
                },
                ensure_ascii=False,
            )
        return json.dumps(data, ensure_ascii=False)
