from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.geekflare_tool.geekflare_base import GeekflareBaseTool


class GeekflareBrandToolSchema(BaseModel):
    """Input schema for GeekflareBrandTool."""

    url: str = Field(..., description="Website URL or domain, e.g. https://stripe.com.")


class GeekflareBrandTool(GeekflareBaseTool):
    """Get a company's brand identity from its website.

    Attributes:
        mode: "standard", or "enriched" to add company intelligence.
    """

    name: str = "Geekflare Brand"
    description: str = (
        "Get a company's brand identity from its website with Geekflare: name, logos, "
        "brand colors, fonts, and social profiles, as JSON. Fields the site doesn't "
        "publish are omitted."
    )
    args_schema: type[BaseModel] = GeekflareBrandToolSchema
    mode: Literal["standard", "enriched"] = Field(
        default="standard", description="Depth of brand data."
    )

    def _run(self, url: str) -> str:
        url = self._validate_url(url)
        return self._post_json("brand", {"url": url, "mode": self.mode})
