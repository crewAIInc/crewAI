"""MrScraper existing-scraper run tools."""

from typing import Any, Literal

from pydantic import BaseModel

from crewai_tools.tools.mrscraper.base import MrScraperBaseTool
from crewai_tools.tools.mrscraper.payloads import existing_run_payload
from crewai_tools.tools.mrscraper.schemas import (
    RunExistingScraperBatchInput,
    RunExistingScraperInput,
)


class MrScraperRunExistingScraperTool(MrScraperBaseTool):
    """Run an existing AI or manual scraper on one URL."""

    name: str = "mrscraper_run_existing_scraper"
    description: str = (
        "Use this to run one URL through an existing AI or manual scraper. Choose the AI "
        "agent type carefully; conditional options are validated before any request."
    )
    args_schema: type[BaseModel] = RunExistingScraperInput

    def _run(self, **values: Any) -> str:
        scraper_type = values["scraper_type"]
        endpoint = (
            "/api/v1/scrapers-manual-rerun"
            if scraper_type == "manual"
            else "/api/v1/scrapers-ai-rerun"
        )
        return self._client.request(
            "POST",
            "primary",
            endpoint,
            json_body=existing_run_payload(values),
        )


class MrScraperRunExistingScraperBatchTool(MrScraperBaseTool):
    """Run an existing AI or manual scraper on a URL batch."""

    name: str = "mrscraper_run_existing_scraper_batch"
    description: str = (
        "Use this potentially expensive batch operation to run multiple URLs through one "
        "existing AI or manual scraper. Use run_existing_scraper for a single URL."
    )
    args_schema: type[BaseModel] = RunExistingScraperBatchInput

    def _run(
        self,
        scraper_type: Literal["ai", "manual"],
        scraper_id: str,
        urls: list[str],
    ) -> str:
        base = (
            "/api/v1/scrapers-manual-rerun"
            if scraper_type == "manual"
            else "/api/v1/scrapers-ai-rerun"
        )
        return self._client.request(
            "POST",
            "primary",
            f"{base}/bulk",
            json_body={"scraperId": scraper_id, "urls": urls},
        )


__all__ = [
    "MrScraperRunExistingScraperBatchTool",
    "MrScraperRunExistingScraperTool",
]
