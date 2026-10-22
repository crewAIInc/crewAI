import json
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


_ACTIONS = ("read", "cite", "connect", "transact", "administer")
_FACT_FIELDS = ("phone", "address", "hours", "service_area")


def _client(api_key: str | None) -> Any:
    try:
        import crawlcheck
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'crawlcheck'. Install with:\n"
            "  uv add crewai-tools --extra crawlcheck\n"
            "or\n"
            "  pip install crawlcheck\n"
        ) from exc
    return crawlcheck.CrawlCheck(key=api_key)


class CrawlCheckSiteCheckToolSchema(BaseModel):
    """Input for CrawlCheckSiteCheckTool."""

    site: str = Field(
        ...,
        description="The website's domain or any URL on it, e.g. example.com or https://example.com/page",
    )
    action: str = Field(
        "read",
        description="What the crew is about to do with the site: read, cite, connect (an API or MCP server), "
        "transact (pay, book, submit personal data) or administer",
    )


class CrawlCheckSiteCheckTool(BaseTool):
    """Ask CrawlCheck before the crew reads, cites, connects to or buys from a website.

    CrawlCheck (https://crawlcheck.io) measures websites the way AI crawlers do and returns a signed
    decision for an action: allow, warn, require_confirmation or block, with the reasons (whether AI
    crawlers are served the page, whether the business still appears to be live, security posture,
    look-alike or lead-generation networks, and more). No API key is needed for decisions; a licence
    key unlocks the full findings for licensed domains.
    """

    name: str = "CrawlCheck site check"
    description: str = (
        "Before acting on a website, get CrawlCheck's signed decision for that action (allow, warn, "
        "require_confirmation or block) with the reasons. Use it before reading, citing, connecting to or "
        "buying from a site you have not checked. The result's `proceed` field is the answer."
    )
    args_schema: type[BaseModel] = CrawlCheckSiteCheckToolSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CRAWLCHECK_API_KEY",
                description="CrawlCheck licence key (optional; unlocks full findings for licensed domains)",
                required=False,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["crawlcheck"])
    api_key: str | None = None
    allow_warn: bool = False
    _tools: Any = None

    def __init__(
        self, api_key: str | None = None, allow_warn: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        import os

        self.api_key = api_key or os.environ.get("CRAWLCHECK_API_KEY")
        self.allow_warn = allow_warn
        import crawlcheck

        cc = _client(self.api_key)
        self._tools = {
            t["name"]: t for t in crawlcheck.agent_tools(cc, allow_warn=allow_warn)
        }

    def _run(self, site: str, action: str = "read", **_: Any) -> str:
        if action not in _ACTIONS:
            action = "read"
        result = self._tools["crawlcheck_check_site"]["execute"](
            {"site": site, "action": action}
        )
        return json.dumps(result)


class CrawlCheckVerifyFactToolSchema(BaseModel):
    """Input for CrawlCheckVerifyFactTool."""

    site: str = Field(..., description="The business's website domain or any URL on it")
    field: str = Field(
        ..., description="Which fact: phone, address, hours or service_area"
    )
    value: str = Field(
        ..., description="The value you hold, e.g. a phone number or a street address"
    )


class CrawlCheckVerifyFactTool(BaseTool):
    """Check one fact the crew holds about a business against what the business itself publishes.

    Returns a signed document saying whether the value matches the site's own phone, address, hours or
    service area, so an agent does not quote a number that is no longer on the business's own page.
    """

    name: str = "CrawlCheck verify fact"
    description: str = (
        "Check one fact you hold about a business (phone, address, hours or service_area) against what "
        "the business itself publishes on its website. Returns whether it matches, signed by CrawlCheck."
    )
    args_schema: type[BaseModel] = CrawlCheckVerifyFactToolSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CRAWLCHECK_API_KEY",
                description="CrawlCheck licence key (optional)",
                required=False,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["crawlcheck"])
    api_key: str | None = None
    _tools: Any = None

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        import os

        self.api_key = api_key or os.environ.get("CRAWLCHECK_API_KEY")
        import crawlcheck

        cc = _client(self.api_key)
        self._tools = {t["name"]: t for t in crawlcheck.agent_tools(cc)}

    def _run(self, site: str, field: str, value: str, **_: Any) -> str:
        if field not in _FACT_FIELDS:
            return json.dumps(
                {
                    "ok": False,
                    "why": ["field must be one of: " + ", ".join(_FACT_FIELDS)],
                }
            )
        result = self._tools["crawlcheck_verify_fact"]["execute"](
            {"site": site, "field": field, "value": value}
        )
        return json.dumps(result)
