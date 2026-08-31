"""MrScraper account tool."""

from pydantic import BaseModel

from crewai_tools.tools.mrscraper.base import MrScraperBaseTool
from crewai_tools.tools.mrscraper.schemas import GetAccountInfoInput


class MrScraperGetAccountInfoTool(MrScraperBaseTool):
    """Retrieve subscription and token usage information."""

    name: str = "mrscraper_get_account_info"
    description: str = (
        "Use this narrow read-only tool to inspect MrScraper account details, "
        "token usage, and token limits. It does not scrape a page or create a job."
    )
    args_schema: type[BaseModel] = GetAccountInfoInput

    def _run(self) -> str:
        return self._client.request("GET", "primary", "/api/v1/subscription-accounts")


__all__ = ["MrScraperGetAccountInfoTool"]
