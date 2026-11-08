import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


ALION_OPEN_URL = "https://alion.io/mcp/open"
ALION_URL = "https://alion.io/mcp"


class AlionBaseTool(BaseTool):
    """Base for the Alion tools: one call to one tool of the Alion MCP server.

    Without ``ALION_API_KEY`` the tools use Alion's free open address (compact company
    and job searches and lookalikes, within a daily allowance per network). With a key
    they use the full server: full company profiles and every filter.
    """

    alion_tool: str = ""
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    timeout: float = 60.0
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ALION_API_KEY",
                description="API key for Alion (optional: the search tools work without one)",
                required=False,
            ),
        ]
    )

    def _key(self) -> str | None:
        return self.api_key or os.getenv("ALION_API_KEY") or None

    def _call(
        self, arguments: dict[str, Any], tool: str | None = None
    ) -> dict[str, Any]:
        key = self._key()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "crewai-tools-alion",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool or self.alion_tool,
                "arguments": {k: v for k, v in arguments.items() if v is not None},
            },
        }
        try:
            response = requests.post(
                ALION_URL if key else ALION_OPEN_URL,
                json=body,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            return {"success": False, "error": f"Alion request failed: {e}"}
        except ValueError:
            return {
                "success": False,
                "error": "Alion returned a response that is not JSON",
            }

        if "error" in data:
            return {
                "success": False,
                "error": data["error"].get("message", str(data["error"])),
            }
        result = data.get("result", {})
        text = "".join(
            c.get("text", "") for c in result.get("content", []) if isinstance(c, dict)
        )
        payload = result.get("structuredContent")
        if payload is None:
            payload = _parse_text(text)
        if result.get("isError"):
            message = None
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("error")
            return {
                "success": False,
                "error": str(message or text or "Alion refused the call"),
            }
        return {"success": True, "result": payload}


def _parse_text(text: str) -> Any:
    """The text content of an MCP result: Alion's JSON when it parses, else the text itself."""
    try:
        return json.loads(text)
    except ValueError:
        return {"text": text}


class AlionCompanySearchSchema(BaseModel):
    query: str | None = Field(
        default=None,
        description="Company name prefix or domain prefix, e.g. 'Acme' or 'acme.'",
    )
    country: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 code or country name of an office, e.g. 'DE'",
    )
    industry: str | None = Field(
        default=None, description="Industry or segment, e.g. 'Fintech', 'Healthcare'"
    )
    size: str | None = Field(
        default=None,
        description="Employee band or comma list: 1-10, 11-50, 51-200, 201-500, 501-1000, 1001-5000, 5000+",
    )
    startup: bool | None = Field(
        default=None, description="True for companies classified as startups"
    )
    funding_stage: str | None = Field(
        default=None,
        description="A round or comma list, e.g. 'seed', 'series_a', or a group: 'early', 'growth', 'venture'",
    )
    raised_since: str | None = Field(
        default=None,
        description="Only companies with a matching round since a date ('2026-01-01') or a window ('90d', '6m')",
    )
    investor: str | None = Field(
        default=None,
        description="An investor by domain or name, e.g. 'sequoiacap.com': its portfolio companies",
    )
    technology: str | None = Field(
        default=None,
        description="A technology required in the company's job postings, e.g. 'Kubernetes'",
    )
    hiring: bool | None = Field(
        default=None, description="True for companies with open roles now"
    )
    sort: (
        Literal["recent_hiring", "recent_funding", "open_jobs", "newest", "name"] | None
    ) = Field(default=None, description="Order of the results")
    limit: int | None = Field(
        default=None, ge=1, le=50, description="Companies to return (1-50, default 20)"
    )


class AlionCompanySearchTool(AlionBaseTool):
    name: str = "Alion Company Search"
    description: str = (
        "Searches Alion's live graph of companies worldwide by industry, country, size, startup status, "
        "funding stage, recent rounds, investor, technology and hiring. Returns name, domain, industry, "
        "size, headquarters country, founding year, newest round and open roles for each company. "
        "Works without an API key."
    )
    args_schema: type[BaseModel] = AlionCompanySearchSchema
    alion_tool: str = "search_companies"

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return self._call({**kwargs, "detail": "compact"})


class AlionCompanyProfileSchema(BaseModel):
    domain: str | None = Field(
        default=None,
        description="Company web domain, e.g. 'stripe.com' (the most exact way to name a company)",
    )
    name: str | None = Field(
        default=None, description="Company name, when the domain is unknown"
    )


class AlionCompanyProfileTool(AlionBaseTool):
    name: str = "Alion Company Profile"
    description: str = (
        "Returns one company in depth from Alion: what it does, funding rounds and investors, hiring and open "
        "roles, tech stack from its job postings, offices and visa sponsorship. Needs ALION_API_KEY; without it, "
        "returns which company a name or domain means."
    )
    args_schema: type[BaseModel] = AlionCompanyProfileSchema
    alion_tool: str = "get_company"

    def _run(
        self, domain: str | None = None, name: str | None = None
    ) -> dict[str, Any]:
        if not domain and not name:
            return {"success": False, "error": "Give a company domain or name"}
        if self._key():
            return self._call({"domain": domain, "name": name})
        resolved = self._call({"domain": domain, "name": name}, tool="resolve_company")
        if resolved.get("success"):
            resolved["note"] = (
                "Identity only: set ALION_API_KEY for the full profile (funding, hiring, stack, offices)."
            )
        return resolved


class AlionSimilarCompaniesSchema(BaseModel):
    company: str = Field(
        ...,
        description="The company to find lookalikes of: domain ('stripe.com') or name",
    )
    country: str | None = Field(
        default=None,
        description="Only lookalikes with an office in this country, e.g. 'GB'",
    )
    limit: int | None = Field(
        default=None, ge=1, le=25, description="Lookalikes to return (1-25, default 10)"
    )


class AlionSimilarCompaniesTool(AlionBaseTool):
    name: str = "Alion Similar Companies"
    description: str = (
        "Finds companies similar to a given company in Alion's company graph (same segment, similar stack and "
        "size), optionally in one country. Works without an API key."
    )
    args_schema: type[BaseModel] = AlionSimilarCompaniesSchema
    alion_tool: str = "similar_companies"

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return self._call({**kwargs, "detail": "compact"})


class AlionJobSearchSchema(BaseModel):
    query: str | None = Field(
        default=None,
        description="Words in the title or description, e.g. 'python backend'",
    )
    role: str | None = Field(
        default=None,
        description="A role or role family, e.g. 'Data Engineer', 'Backend'",
    )
    company: str | None = Field(
        default=None,
        description="One employer by domain or exact name, e.g. 'stripe.com'",
    )
    country: str | None = Field(
        default=None, description="ISO 3166-1 alpha-2 code or country name, e.g. 'PL'"
    )
    city: str | None = Field(default=None, description="City in English, e.g. 'Berlin'")
    work_mode: Literal["on_site", "hybrid", "remote"] | None = Field(
        default=None, description="Where the work happens"
    )
    seniority: (
        Literal["intern", "junior", "middle", "senior", "staff", "lead", "head"] | None
    ) = Field(default=None, description="Seniority of the role")
    technology: str | None = Field(
        default=None, description="A required technology, e.g. 'Go'"
    )
    salary_min_usd: int | None = Field(
        default=None, ge=0, description="Minimum annual pay in USD"
    )
    posted_since: str | None = Field(
        default=None,
        description="Only postings first seen after a date ('2026-09-01') or a window ('7d')",
    )
    visa_sponsorship: bool | None = Field(
        default=None, description="True for postings that state visa sponsorship"
    )
    limit: int | None = Field(
        default=None, ge=1, le=50, description="Postings to return (1-50, default 20)"
    )


class AlionJobSearchTool(AlionBaseTool):
    name: str = "Alion Job Search"
    description: str = (
        "Searches verified job postings in Alion by role, company, country, city, work mode, seniority, "
        "technology, pay and visa sponsorship, newest first. Useful to see what a company or a market is "
        "hiring for. Works without an API key."
    )
    args_schema: type[BaseModel] = AlionJobSearchSchema
    alion_tool: str = "search_jobs"

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return self._call({**kwargs, "detail": "compact"})
