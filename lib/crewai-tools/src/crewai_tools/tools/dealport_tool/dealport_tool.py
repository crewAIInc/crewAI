from __future__ import annotations

import json
import os
from typing import Any, Literal
import urllib.parse

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


DEFAULT_DEALPORT_BASE_URL = "https://app.dealport.com"
VALID_NYC_BOROUGHS = frozenset({"MN", "BK", "QN", "BX", "SI"})

DealportAction = Literal[
    "preview_companies",
    "search_companies",
    "search_buildings",
    "get_record",
    "find_owner",
]


class DealportToolInput(BaseModel):
    """Input schema for querying Dealport V4 company, building, and owner endpoints."""

    action: DealportAction = Field(
        default="preview_companies",
        description=(
            "Dealport V4 operation to execute: "
            "'preview_companies' ($0.00 US company market sizing count and up to 5 samples), "
            "'search_companies' ($0.01/company Record, $0.00 repeat within 30 days), "
            "'search_buildings' ($0.01/NYC building Record, $0.00 repeat within 30 days), "
            "'get_record' ($0.01 cited company UUID or 10-digit NYC BBL Record), or "
            "'find_owner' ($0.09 verified human owner/contact hit, $0.00 miss)."
        ),
    )
    q: str | None = Field(
        default=None,
        description="Company name prefix (for company queries) or building address/owner filter.",
    )
    state: str | None = Field(
        default=None,
        description="Two-letter US state code for company queries, e.g. 'OH' or 'TX'.",
    )
    cities: list[str] | None = Field(
        default=None,
        description="Optional city names within state, e.g. ['Columbus', 'Dayton'].",
    )
    counties: list[str] | None = Field(
        default=None,
        description="Optional county names within state, e.g. ['Franklin', 'Montgomery'].",
    )
    naics: list[str] | None = Field(
        default=None,
        description="2-to-6 digit NAICS industry codes, e.g. ['238220', '332710'].",
    )
    domain: str | None = Field(
        default=None,
        description="Company website domain, e.g. 'example.com'.",
    )
    min_revenue: float | None = Field(
        default=None,
        ge=0,
        description="Minimum annual revenue in USD.",
    )
    max_revenue: float | None = Field(
        default=None,
        ge=0,
        description="Maximum annual revenue in USD.",
    )
    min_employees: int | None = Field(
        default=None,
        ge=0,
        description="Minimum employee headcount.",
    )
    max_employees: int | None = Field(
        default=None,
        ge=0,
        description="Maximum employee headcount.",
    )
    independent: bool | None = Field(
        default=None,
        description="When True, restrict to privately held independent companies.",
    )
    licensed: bool | None = Field(
        default=None,
        description="When True, restrict to companies holding an active state license.",
    )
    owner_contact: bool | None = Field(
        default=None,
        description="When True, restrict to companies with an owner phone or email on file.",
    )
    borough: str | None = Field(
        default=None,
        description="NYC borough code required for 'search_buildings': MN, BK, QN, BX, or SI.",
    )
    min_units: int | None = Field(
        default=None,
        ge=1,
        description="Minimum unit count for 'search_buildings'.",
    )
    loan_due_by: int | None = Field(
        default=None,
        description="Maximum ACRIS mortgage maturity year for 'search_buildings', e.g. 2027.",
    )
    loan_holder: str | None = Field(
        default=None,
        description="Mortgage holder category or lender filter for 'search_buildings'.",
    )
    signal: str | None = Field(
        default=None,
        description="Optional building signal filter for 'search_buildings'.",
    )
    record_id: str | None = Field(
        default=None,
        description="Company UUID or 10-digit NYC BBL required for 'get_record'.",
    )
    company_id: str | None = Field(
        default=None,
        description="Dealport company UUID required for 'find_owner'.",
    )
    person: str | None = Field(
        default=None,
        description="Optional full name of a specific principal for 'find_owner'.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum number of Records to return (1 to 50).",
    )
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor from a previous search response.",
    )


def _clean_record_segment(raw_id: str) -> str:
    cleaned = raw_id.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError(
            "record_id must be a single company UUID or 10-digit BBL path segment."
        )
    return urllib.parse.quote(cleaned, safe="")


def _build_company_filters(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(params.get("q"), str) and params["q"].strip():
        out["q"] = params["q"].strip()
    if isinstance(params.get("state"), str) and params["state"].strip():
        out["state"] = params["state"].strip().upper()
    if isinstance(params.get("domain"), str) and params["domain"].strip():
        out["domain"] = params["domain"].strip()

    for list_key in ("cities", "counties", "naics"):
        raw_items = params.get(list_key)
        if isinstance(raw_items, list):
            cleaned_items = [
                str(item).strip() for item in raw_items if str(item).strip()
            ]
            if cleaned_items:
                out[list_key] = cleaned_items

    for num_key in (
        "min_revenue",
        "max_revenue",
        "min_employees",
        "max_employees",
        "limit",
    ):
        val = params.get(num_key)
        if val is not None and not isinstance(val, bool):
            out[num_key] = val

    if isinstance(params.get("cursor"), str) and params["cursor"].strip():
        out["cursor"] = params["cursor"].strip()

    for bool_key in ("independent", "licensed", "owner_contact"):
        if params.get(bool_key) is True:
            out[bool_key] = True

    return out


def _filters_to_query_string(filters: dict[str, Any]) -> str:
    pairs: list[tuple[str, str]] = []
    for key, value in filters.items():
        if value is True:
            pairs.append((key, "1"))
        elif isinstance(value, list):
            joined = ",".join(str(v) for v in value if str(v))
            if joined:
                pairs.append((key, joined))
        elif value is not None:
            pairs.append((key, str(value)))
    return urllib.parse.urlencode(pairs)


class DealportTool(BaseTool):
    """Query Dealport V4 cited US private company Records, commercial buildings, and owners."""

    name: str = "Dealport SMB Intelligence Tool"
    description: str = (
        "Query Dealport's 28M+ cited US private company Records, NYC commercial and "
        "multifamily building Records, free ($0.00) market sizing counts, and verified "
        "human owner contacts backed by primary public filings."
    )
    args_schema: type[BaseModel] = DealportToolInput

    api_key: str | None = Field(
        default=None,
        description=(
            "Dealport API key (dp_live_...). Falls back to DP_API_KEY or DEALPORT_API_KEY."
        ),
        exclude=True,
    )
    base_url: str = Field(
        default=DEFAULT_DEALPORT_BASE_URL,
        description="Base URL for the Dealport API.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="HTTP request timeout in seconds.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="DP_API_KEY",
                description="Dealport workspace API key (dp_live_...)",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    def _resolve_api_key(self) -> str:
        key = (
            self.api_key
            or os.environ.get("DP_API_KEY")
            or os.environ.get("DEALPORT_API_KEY")
            or ""
        ).strip()
        if not key:
            raise ValueError(
                "DP_API_KEY (or DEALPORT_API_KEY) environment variable or api_key parameter is required."
            )
        return key

    def _resolve_base_url(self) -> str:
        base = self.base_url.strip().rstrip("/")
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("base_url must be a valid http(s) URL.")
        return base

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
    ) -> str:
        api_key = self._resolve_api_key()
        url = f"{self._resolve_base_url()}{path}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "crewai-tools-dealport/1.0",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        resp = requests.request(
            method=method,
            url=url,
            json=json_body,
            headers=headers,
            timeout=self.timeout,
        )
        if resp.status_code >= 300:
            snippet = resp.text[:300].strip()
            return f"Dealport API error ({resp.status_code}): {snippet}"
        if not resp.text:
            return "{}"
        try:
            return json.dumps(resp.json(), ensure_ascii=False)
        except ValueError:
            return resp.text

    def _run(
        self,
        action: DealportAction = "preview_companies",
        **kwargs: Any,
    ) -> str:
        try:
            if action == "preview_companies":
                filters = _build_company_filters(kwargs)
                filters.pop("cursor", None)
                return self._request(
                    "/api/v1/companies/preview",
                    method="POST",
                    json_body=filters,
                )

            if action == "search_companies":
                filters = _build_company_filters(kwargs)
                qs = _filters_to_query_string(filters)
                suffix = f"?{qs}" if qs else ""
                return self._request(f"/api/v1/companies{suffix}")

            if action == "search_buildings":
                raw_borough = kwargs.get("borough")
                if not isinstance(raw_borough, str) or not raw_borough.strip():
                    return "Error: borough is required for search_buildings (MN, BK, QN, BX, or SI)."
                borough = raw_borough.strip().upper()
                if borough not in VALID_NYC_BOROUGHS:
                    return "Error: borough must be one of MN, BK, QN, BX, or SI."
                query: list[tuple[str, str]] = [("borough", borough)]
                if kwargs.get("min_units") is not None:
                    query.append(("min_units", str(kwargs["min_units"])))
                if isinstance(kwargs.get("q"), str) and kwargs["q"].strip():
                    query.append(("q", kwargs["q"].strip()))
                if kwargs.get("loan_due_by") is not None:
                    query.append(("loan_due_by", str(kwargs["loan_due_by"])))
                if (
                    isinstance(kwargs.get("loan_holder"), str)
                    and kwargs["loan_holder"].strip()
                ):
                    query.append(("loan_holder", kwargs["loan_holder"].strip()))
                if isinstance(kwargs.get("signal"), str) and kwargs["signal"].strip():
                    query.append(("signal", kwargs["signal"].strip()))
                if kwargs.get("limit") is not None:
                    query.append(("limit", str(kwargs["limit"])))
                if isinstance(kwargs.get("cursor"), str) and kwargs["cursor"].strip():
                    query.append(("cursor", kwargs["cursor"].strip()))
                return self._request(
                    f"/api/v1/buildings?{urllib.parse.urlencode(query)}"
                )

            if action == "get_record":
                raw_record_id = kwargs.get("record_id")
                if not isinstance(raw_record_id, str) or not raw_record_id.strip():
                    return "Error: record_id is required for get_record."
                encoded = _clean_record_segment(raw_record_id)
                return self._request(f"/api/v1/records/{encoded}")

            if action == "find_owner":
                raw_company_id = kwargs.get("company_id")
                if not isinstance(raw_company_id, str) or not raw_company_id.strip():
                    return "Error: company_id is required for find_owner."
                body: dict[str, Any] = {"company_id": raw_company_id.strip()}
                if isinstance(kwargs.get("person"), str) and kwargs["person"].strip():
                    body["person"] = kwargs["person"].strip()
                return self._request("/api/v1/owners", method="POST", json_body=body)

            return f"Error: Unsupported Dealport action: {action}"
        except requests.Timeout:
            return "Dealport API request timed out. Please retry."
        except ValueError as err:
            return f"Error: {err}"
        except requests.RequestException as err:
            return f"Dealport API network error: {err}"
