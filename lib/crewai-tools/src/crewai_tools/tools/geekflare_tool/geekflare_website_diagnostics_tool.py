from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.geekflare_tool.geekflare_base import GeekflareBaseTool


DiagnosticCheck = Literal[
    "site_status",
    "dns_records",
    "tls_scan",
    "redirect_check",
    "broken_links",
    "load_time",
    "meta_tags",
    "lighthouse",
]

# check -> (endpoint path, whether it accepts followRedirect)
_CHECKS: dict[str, tuple[str, bool]] = {
    "site_status": ("up", True),
    "dns_records": ("dnsrecord", False),
    "tls_scan": ("tlsscan", False),
    "redirect_check": ("redirectcheck", False),
    "broken_links": ("brokenlink", True),
    "load_time": ("loadtime", True),
    "meta_tags": ("metascraping", False),
    "lighthouse": ("lighthouse", True),
}


class GeekflareWebsiteDiagnosticsToolSchema(BaseModel):
    """Input schema for GeekflareWebsiteDiagnosticsTool."""

    url: str = Field(
        ..., description="Website URL or domain, e.g. https://example.com."
    )
    check: DiagnosticCheck = Field(
        ...,
        description=(
            "Which check to run: site_status (up/down and HTTP status), dns_records, "
            "tls_scan (certificate and TLS versions), redirect_check, broken_links, "
            "load_time, meta_tags (title, description, Open Graph), or lighthouse "
            "(performance/SEO/accessibility scores; slow)."
        ),
    )


class GeekflareWebsiteDiagnosticsTool(GeekflareBaseTool):
    """Run website diagnostics with Geekflare.

    Attributes:
        follow_redirect: Follow redirects before checking, where supported.
        lighthouse_device: Device for Lighthouse audits.
    """

    name: str = "Geekflare Website Diagnostics"
    description: str = (
        "Run a website check with Geekflare and return the result as JSON: uptime, "
        "DNS records, TLS/SSL certificate, redirect chain, broken links, load time, "
        "meta tags, or a Lighthouse audit. Reports findings only; it does not change "
        "the website."
    )
    args_schema: type[BaseModel] = GeekflareWebsiteDiagnosticsToolSchema
    follow_redirect: bool = Field(
        default=False, description="Follow redirects before checking."
    )
    lighthouse_device: Literal["desktop", "mobile"] = Field(
        default="desktop", description="Device for Lighthouse audits."
    )

    def _run(self, url: str, check: DiagnosticCheck) -> str:
        url = self._validate_url(url)
        path, supports_redirect = _CHECKS[check]
        payload: dict[str, object] = {"url": url}
        if supports_redirect:
            payload["followRedirect"] = self.follow_redirect
        if check == "lighthouse":
            payload["device"] = self.lighthouse_device
        return self._post_json(path, payload)
