from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.geekflare_tool.geekflare_base import GeekflareBaseTool


class GeekflareScreenshotToolSchema(BaseModel):
    """Input schema for GeekflareScreenshotTool."""

    url: str = Field(..., description="Full URL of the page to capture.")


class GeekflareScreenshotTool(GeekflareBaseTool):
    """Capture a screenshot of a web page and return a hosted image URL.

    Attributes:
        full_page: Capture the whole page instead of the viewport.
        device: Device to emulate.
        image_type: png, jpeg, or webp.
        selector: CSS selector of a single element to capture.
    """

    name: str = "Geekflare Screenshot"
    description: str = (
        "Take a screenshot of a web page with Geekflare and return JSON that includes "
        "a hosted image URL."
    )
    args_schema: type[BaseModel] = GeekflareScreenshotToolSchema
    full_page: bool = Field(default=False, description="Capture the full page.")
    device: Literal["desktop", "mobile"] = Field(
        default="desktop", description="Device to emulate."
    )
    image_type: Literal["png", "jpeg", "webp"] = Field(
        default="png", description="Image format."
    )
    selector: str | None = Field(
        default=None, description="CSS selector of one element to capture."
    )

    def _run(self, url: str) -> str:
        url = self._validate_url(url)
        return self._post_json(
            "screenshot",
            {
                "url": url,
                "fullPage": self.full_page,
                "device": self.device,
                "type": self.image_type,
                "selector": self.selector,
            },
        )
