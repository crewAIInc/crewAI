from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.geekflare_tool.geekflare_base import GeekflareBaseTool


ScrapeFormat = Literal[
    "markdown", "markdown-llm", "html", "html-llm", "text", "text-llm", "json"
]


def _default_formats() -> list[ScrapeFormat]:
    return ["markdown"]


class GeekflareScrapeWebsiteToolSchema(BaseModel):
    """Input schema for GeekflareScrapeWebsiteTool."""

    url: str = Field(..., description="Full URL of the public page to scrape.")


class GeekflareScrapeWebsiteTool(GeekflareBaseTool):
    """Scrape a public web page into Markdown, text, HTML, or structured JSON.

    Attributes:
        formats: Output format(s), up to 3. The *-llm variants strip boilerplate.
        render_js: Force JavaScript rendering on or off; None decides automatically.
        wait_time: Seconds to wait after page load before capturing.
        device: Device to emulate.
        template: Ready-made structured extraction: "product" (title, price,
            availability, images, ratings) or "contact" (emails, phones, socials).
    """

    name: str = "Geekflare Scrape Website"
    description: str = (
        "Fetch a public web page with Geekflare and return its content as JSON "
        "(Markdown by default). Only for publicly accessible pages."
    )
    args_schema: type[BaseModel] = GeekflareScrapeWebsiteToolSchema
    formats: list[ScrapeFormat] = Field(
        default_factory=_default_formats, description="Output format(s)."
    )
    render_js: bool | None = Field(default=None, description="Force JS rendering.")
    wait_time: float | None = Field(
        default=None, ge=0, description="Seconds to wait after load."
    )
    device: Literal["desktop", "mobile"] = Field(
        default="desktop", description="Device to emulate."
    )
    template: Literal["product", "contact"] | None = Field(
        default=None, description="Structured extraction template."
    )

    def _run(self, url: str) -> str:
        url = self._validate_url(url)
        return self._post_json(
            "webscraping",
            {
                "url": url,
                "format": self.formats,
                "renderJS": self.render_js,
                "waitTime": self.wait_time,
                "device": self.device,
                "template": self.template,
                "extractionMode": "template" if self.template else None,
            },
        )
