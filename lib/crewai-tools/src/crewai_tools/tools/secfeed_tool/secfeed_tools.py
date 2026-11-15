"""SEC EDGAR tools backed by the SecFeed API (https://secfeedapi.com).

Each tool returns compact plain text sized for an LLM context window, with a
link to the source filing so agents can cite what they read.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr
import requests


BASE_URL = "https://secfeedapi.com"
MAX_CHARS = 6000

# Which item of a filing answers which kind of question.
SECTIONS: dict[str, dict[str, str]] = {
    "10-K": {"business": "1", "risk": "1A", "mdna": "7", "market_risk": "7A"},
    "10-Q": {
        "business": "part1item2",
        "risk": "part2item1a",
        "mdna": "part1item2",
        "market_risk": "part1item3",
    },
}
SECTION_NAMES = {
    "business": "Business",
    "risk": "Risk Factors",
    "mdna": "Management's Discussion and Analysis",
    "market_risk": "Market Risk",
}
ROUTES: list[tuple[tuple[str, ...], str]] = [
    (("risk", "threat", "uncertain", "litigation", "regulat", "lawsuit"), "risk"),
    (
        ("interest rate", "currency", "foreign exchange", "hedg", "market risk"),
        "market_risk",
    ),
    (
        ("product", "segment", "customer", "competit", "employee", "strategy"),
        "business",
    ),
]
STOP = set(
    "the a an of to in and or for on is are was were be by with what how did does "
    "do its it this that from as at which who their they has have had last year "
    "quarter company".split()
)
CODES = {
    "P": "open market buy",
    "S": "open market sale",
    "A": "grant",
    "M": "option exercise",
    "F": "tax withholding",
    "G": "gift",
}


def pick_section(question: str) -> str:
    """Section key for a question: risk, market_risk, business, else mdna."""
    q = question.lower()
    for words, section in ROUTES:
        if any(w in q for w in words):
            return section
    return "mdna"


def _terms(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z][a-z0-9&\-]+", text.lower())
        if w not in STOP and len(w) > 2
    }


def best_passages(text: str, question: str, max_chars: int = MAX_CHARS) -> str:
    """Paragraphs ranked by shared words with the question, kept in document order."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80]
    if not paras:
        return text[:max_chars]
    terms = _terms(question)
    order = sorted(range(len(paras)), key=lambda i: -len(terms & _terms(paras[i])))
    keep: set[int] = set()
    size = 0
    for i in order:
        if size + len(paras[i]) > max_chars and keep:
            continue
        keep.add(i)
        size += len(paras[i])
        if size >= max_chars:
            break
    return "\n\n".join(paras[i] for i in sorted(keep))


def money(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.2f}"


class _SecFeedBaseTool(BaseTool):
    """Shared HTTP client and API key handling."""

    api_key: str | None = Field(default=None, description="SecFeed API key")
    base_url: str = Field(default=BASE_URL, description="SecFeed API base URL")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SECFEED_API_KEY",
                description="API key for SecFeed (https://secfeedapi.com/signup/)",
                required=True,
            ),
        ]
    )
    _filings: dict[tuple[str, str], dict[str, Any] | None] = PrivateAttr(
        default_factory=dict
    )

    def _key(self) -> str:
        key = self.api_key or os.getenv("SECFEED_API_KEY")
        if not key:
            raise ValueError(
                "SECFEED_API_KEY is not set. Get a key at https://secfeedapi.com/signup/"
            )
        return key

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        raw: bool = False,
    ) -> Any:
        response = requests.request(
            method,
            f"{self.base_url.rstrip('/')}{path}",
            params={k: v for k, v in (params or {}).items() if v is not None},
            json=body,
            headers={"Authorization": f"Bearer {self._key()}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text if raw else response.json()

    def _latest_filing(self, ticker: str, form: str) -> dict[str, Any] | None:
        key = (ticker.upper(), form)
        if key not in self._filings:
            res = self._request(
                "POST",
                "/v1/filings",
                body={
                    "query": f'ticker:{ticker.upper()} AND formType:"{form}"',
                    "from": 0,
                    "size": 1,
                    "sort": "desc",
                },
            )
            self._filings[key] = (res.get("filings") or [None])[0]
        return self._filings[key]


class SecFeedFilingSearchSchema(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    question: str = Field(..., description="What to find out from the filing")
    form: Literal["10-K", "10-Q"] = Field(
        default="10-K",
        description="10-K for the latest annual report, 10-Q for the latest quarter",
    )
    section: str | None = Field(
        default=None,
        description="Optional: business, risk, mdna or market_risk. Picked from the question when empty.",
    )


class SecFeedFilingSearchTool(_SecFeedBaseTool):
    """Answers a question from the right section of a company's latest 10-K or 10-Q."""

    name: str = "SecFeed SEC filing search"
    description: str = (
        "Reads the right section (Business, Risk Factors, MD&A or Market Risk) of a "
        "company's latest 10-K or 10-Q and returns the passages most relevant to the "
        "question, with a link to the filing."
    )
    args_schema: type[BaseModel] = SecFeedFilingSearchSchema
    _sections: dict[tuple[str, str], str] = PrivateAttr(default_factory=dict)

    def _run(
        self,
        ticker: str,
        question: str,
        form: str = "10-K",
        section: str | None = None,
    ) -> str:
        form = form.upper()
        if form not in SECTIONS:
            return "form must be 10-K or 10-Q"
        filing = self._latest_filing(ticker, form)
        if not filing:
            return f"No {form} found for {ticker}. Check the ticker."
        section = section if section in SECTIONS[form] else pick_section(question)
        url, item = filing["linkToFilingDetails"], SECTIONS[form][section]
        try:
            if (url, item) not in self._sections:
                self._sections[(url, item)] = self._request(
                    "GET",
                    "/v1/extractor",
                    {"url": url, "item": item, "type": "text"},
                    raw=True,
                )
        except requests.HTTPError as e:
            return f"Could not read {SECTION_NAMES[section]} of {ticker.upper()} {form}: {e}"
        head = (
            f"{filing['companyName']} ({ticker.upper()}) {form} filed {filing['filedAt']}, "
            f"section: {SECTION_NAMES[section]}\nSource: {url}\n\n"
        )
        return head + best_passages(self._sections[(url, item)], question)


class SecFeedFinancialStatementsSchema(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    form: Literal["10-K", "10-Q"] = Field(
        default="10-K",
        description="10-K for annual figures, 10-Q for the latest quarter",
    )


class SecFeedFinancialStatementsTool(_SecFeedBaseTool):
    """Income statement, balance sheet and cash flow from XBRL."""

    name: str = "SecFeed financial statements"
    description: str = (
        "Income statement, balance sheet and cash flow figures from a company's latest "
        "10-K or 10-Q, taken from the filing's XBRL data, latest periods side by side."
    )
    args_schema: type[BaseModel] = SecFeedFinancialStatementsSchema

    def _run(self, ticker: str, form: str = "10-K") -> str:
        rows = self._request(
            "GET",
            "/v1/financial-statements",
            {"ticker": ticker.upper(), "formType": form.upper(), "n": 1},
        )
        if not rows:
            return f"No XBRL financial statements found for {ticker}."
        data = rows[0]
        out = [
            f"{ticker.upper()} {form.upper()} financial statements "
            f"(accession {data.get('accessionNo')}), USD unless noted"
        ]
        for statement, items in data.items():
            if not isinstance(items, dict):
                continue
            out.append(f"\n{statement}")
            for concept, facts in list(items.items())[:25]:
                if not isinstance(facts, list):
                    continue
                values = []
                for fact in [f for f in facts if not f.get("segment")][:2]:
                    period = fact.get("period") or {}
                    when = period.get("endDate") or period.get("instant") or ""
                    values.append(f"{when}: {money(fact.get('value'))}")
                if values:
                    out.append(f"  {concept}: " + "; ".join(values))
        return "\n".join(out)[:12000]


class SecFeedInsiderTradesSchema(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    code: str | None = Field(
        default=None,
        description="P for open market buys only, S for sales only, empty for all",
    )
    limit: int = Field(default=20, description="How many transactions, up to 50")


class SecFeedInsiderTradesTool(_SecFeedBaseTool):
    """Latest Form 4 insider transactions for a company."""

    name: str = "SecFeed insider trades"
    description: str = (
        "Latest Form 4 insider transactions for a company: role, buy or sell, shares, "
        "price, value and whether the trade was under a 10b5-1 plan."
    )
    args_schema: type[BaseModel] = SecFeedInsiderTradesSchema

    def _run(self, ticker: str, code: str | None = None, limit: int = 20) -> str:
        code = (code or "").strip().upper()[:1] or None
        res = self._request(
            "GET",
            "/v1/insider-trading",
            {
                "ticker": ticker.upper(),
                "code": code if code in ("P", "S") else None,
                "size": max(1, min(int(limit), 50)),
            },
        )
        trades = res.get("transactions") or []
        if not trades:
            return f"No insider transactions found for {ticker}."
        out = [
            f"{ticker.upper()} insider transactions "
            f"({res.get('total')} on file, latest {len(trades)}):"
        ]
        for t in trades:
            owner = (t.get("owners") or [{}])[0]
            role = owner.get("officerTitle") or (
                "Director"
                if owner.get("isDirector")
                else "10% owner"
                if owner.get("isTenPercentOwner")
                else ""
            )
            value = (
                f" = ${money(t['shares'] * t['price'])}"
                if t.get("shares") and t.get("price")
                else ""
            )
            plan = ", 10b5-1 plan" if t.get("aff10b5one") else ""
            out.append(
                f"- {t.get('tx_date')} {owner.get('name')} ({role}): "
                f"{CODES.get(t.get('code'), t.get('code'))} {money(t.get('shares'))} sh "
                f"@ {t.get('price')}{value}{plan}"
            )
        return "\n".join(out)


class SecFeed8KReportsSchema(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    limit: int = Field(default=10, description="How many reports, up to 50")


class SecFeed8KReportsTool(_SecFeedBaseTool):
    """Latest 8-K current reports and the items each one covers."""

    name: str = "SecFeed 8-K reports"
    description: str = (
        "A company's latest 8-K current reports and what each covers (earnings "
        "releases, material agreements, leadership changes, other events), with links."
    )
    args_schema: type[BaseModel] = SecFeed8KReportsSchema

    def _run(self, ticker: str, limit: int = 10) -> str:
        filing = self._latest_filing(ticker, "8-K")
        if not filing:
            return f"No 8-K found for {ticker}."
        res = self._request(
            "GET",
            "/v1/form-8k",
            {"cik": filing["cik"], "size": max(1, min(int(limit), 50))},
        )
        out = [f"{filing['companyName']} ({ticker.upper()}) latest 8-K reports:"]
        for report in res.get("filings") or []:
            items = "; ".join(
                f"{i['item']} {i['title']}" for i in report.get("items") or []
            )
            out.append(f"- {report.get('filed')}: {items}  {report.get('url')}")
        return "\n".join(out)
