"""MrScraper result retrieval tools."""

from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel

from crewai_tools.tools.mrscraper.base import MrScraperBaseTool
from crewai_tools.tools.mrscraper.schemas import (
    GetLatestResultsInput,
    GetResultDetailInput,
    GetResultsInput,
)


class MrScraperGetResultsTool(MrScraperBaseTool):
    """Retrieve a configurable page of results."""

    name: str = "mrscraper_get_results"
    description: str = (
        "Use this to page through results for one scraper with explicit paging and sort "
        "controls. Use get_latest_results when only the newest N records are needed."
    )
    args_schema: type[BaseModel] = GetResultsInput

    def _run(
        self,
        scraper_id: str,
        page: int = 1,
        page_size: int = 10,
        sort_by: Literal["createdAt"] = "createdAt",
        sort_order: Literal["ASC", "DESC"] = "DESC",
    ) -> str:
        return self._client.request(
            "GET",
            "primary",
            "/api/v1/results",
            params={
                "filters[scraperId]": scraper_id,
                "page": page,
                "pageSize": page_size,
                "sort": sort_by,
                "sortOrder": sort_order,
            },
        )


class MrScraperGetLatestResultsTool(MrScraperBaseTool):
    """Retrieve the newest N results."""

    name: str = "mrscraper_get_latest_results"
    description: str = (
        "Use this shortcut for the newest N results from one scraper. Use get_results "
        "instead when page navigation or ascending order is required."
    )
    args_schema: type[BaseModel] = GetLatestResultsInput

    def _run(self, scraper_id: str, count: int = 10) -> str:
        return self._client.request(
            "GET",
            "primary",
            "/api/v1/results",
            params={
                "filters[scraperId]": scraper_id,
                "page": 1,
                "pageSize": count,
                "sort": "createdAt",
                "sortOrder": "DESC",
            },
        )


class MrScraperGetResultDetailTool(MrScraperBaseTool):
    """Retrieve one result by ID."""

    name: str = "mrscraper_get_result_detail"
    description: str = (
        "Use this narrow lookup when a specific MrScraper result ID is already known. "
        "It returns that record rather than a paginated collection."
    )
    args_schema: type[BaseModel] = GetResultDetailInput

    def _run(self, result_id: str) -> str:
        encoded_id = quote(result_id, safe="")
        return self._client.request("GET", "primary", f"/api/v1/results/{encoded_id}")


__all__ = [
    "MrScraperGetLatestResultsTool",
    "MrScraperGetResultDetailTool",
    "MrScraperGetResultsTool",
]
