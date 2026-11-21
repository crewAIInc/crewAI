from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.decodo_base_tool.decodo_base_tool import DecodoBaseTool


__all__ = ["DecodoWebScraperArgs", "DecodoWebScraperTool"]


class DecodoWebScraperArgs(BaseModel):
    url: str = Field(description="Full URL of the page to scrape, including https://")
    geo: str | None = Field(
        default=None,
        description="Country to send the request from, as a country name, e.g. 'United States'",
    )
    js_render: bool = Field(
        default=False,
        description="Open the page in a headless browser. Only needed for pages that load content with JavaScript.",
    )


class DecodoWebScraperTool(DecodoBaseTool):
    """Scrape any public web page with the Decodo Web Scraping API and return it as Markdown.

    Args:
        api_key: Decodo Web Scraping API key. Defaults to the ``DECODO_API_KEY`` environment variable.
        max_chars: Markdown longer than this is truncated, to keep the page within the agent's context.
    """

    name: str = "Decodo Web Scraper"
    description: str = (
        "Fetch any public web page and return its content as Markdown. "
        "Use it to read a specific URL: an article, documentation, a product page."
    )
    args_schema: type[BaseModel] = DecodoWebScraperArgs
    max_chars: int = 45_000

    def _run(
        self, url: str, geo: str | None = None, js_render: bool = False
    ) -> str | ToolFailure:
        from decodo import UniversalParams

        content = self._scrape(
            UniversalParams(
                url=url,
                geo=geo or None,
                headless="html" if js_render else None,
                markdown=True,
            ),
        )
        if isinstance(content, ToolFailure):
            return content

        markdown = str(content)
        if len(markdown) > self.max_chars:
            return f"{markdown[: self.max_chars]}\n\n[Truncated to {self.max_chars} of {len(markdown)} characters.]"
        return markdown
